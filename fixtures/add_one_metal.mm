/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Test fixture: add_one_metal
// Objective-C++ wrapper for Metal kernel execution

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <dlfcn.h>

namespace tvmffi_test_fixtures {

/*! \brief Find the path to the compiled .metallib file
 * 
 * Tries multiple locations:
 * 1. Same directory as the current dylib
 * 2. Current working directory
 * 3. Build directory relative paths
 */
static NSString* findMetallibPath() {
    NSFileManager* fileManager = [NSFileManager defaultManager];
    NSString* cwd = [fileManager currentDirectoryPath];
    
    // Try to get the path of the current dylib
    Dl_info info;
    if (dladdr((void*)findMetallibPath, &info) != 0 && info.dli_fname != nullptr) {
        NSString* dylibPath = [NSString stringWithUTF8String:info.dli_fname];
        NSString* dylibDir = [[dylibPath stringByDeletingLastPathComponent] stringByResolvingSymlinksInPath];
        NSString* metallibPath = [dylibDir stringByAppendingPathComponent:@"add_one_metal.metallib"];
        
        if ([fileManager fileExistsAtPath:metallibPath]) {
            return metallibPath;
        }
        
        // Also try fixtures subdirectory
        NSString* fixturesPath = [dylibDir stringByAppendingPathComponent:@"fixtures/add_one_metal.metallib"];
        if ([fileManager fileExistsAtPath:fixturesPath]) {
            return fixturesPath;
        }
    }
    
    // Try current working directory (convert to absolute path)
    NSString* cwdMetallib = [cwd stringByAppendingPathComponent:@"add_one_metal.metallib"];
    cwdMetallib = [cwdMetallib stringByResolvingSymlinksInPath];
    if ([fileManager fileExistsAtPath:cwdMetallib]) {
        return cwdMetallib;
    }
    
    // Try build directory relative paths (common locations) - convert to absolute paths
    NSArray* searchPaths = @[
        @"build/add_one_metal.metallib",
        @"../build/add_one_metal.metallib",
        @"build/fixtures/add_one_metal.metallib",
        @"../build/fixtures/add_one_metal.metallib"
    ];
    
    for (NSString* path in searchPaths) {
        NSString* absPath = [cwd stringByAppendingPathComponent:path];
        absPath = [absPath stringByResolvingSymlinksInPath];
        if ([fileManager fileExistsAtPath:absPath]) {
            return absPath;
        }
    }
    
    return nil;
}

/*! \brief Perform N-D add one: y = x + 1 (float32, stride-aware) using Metal
 * 
 * This Metal version correctly handles:
 * - Any number of dimensions (1D, 2D, 3D, ...)
 * - Non-contiguous tensors with arbitrary strides
 * - Slices and strided views in any dimension
 * - Zero-copy device memory access
 */
void AddOneMetal(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
    @autoreleasepool {
        int ndim = x.ndim();
        int64_t total_elements = x.numel();
        
        // Get device pointers (Metal buffers)
        void* x_data = x.data_ptr();
        void* y_data = y.data_ptr();
        
        // Get strides and shape
        auto x_strides_view = x.strides();
        auto y_strides_view = y.strides();
        auto shape_view = x.shape();
        
        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Metal device not available");
        }
        
        // Create Metal buffers from existing GPU memory
        // Note: We assume the data pointers are already Metal buffers
        // In a real implementation, you'd need to get the MTLBuffer from the pointer
        // For now, we'll create new buffers and copy data
        
        // Create command queue
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
        
        // Load Metal library from precompiled .metallib file
        NSError* error = nil;
        id<MTLLibrary> library = nil;
        
        // Find the .metallib file
        NSString* metallibPath = findMetallibPath();
        if (metallibPath) {
            // Ensure we have an absolute path
            if (![metallibPath isAbsolutePath]) {
                NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
                metallibPath = [cwd stringByAppendingPathComponent:metallibPath];
            }
            metallibPath = [metallibPath stringByResolvingSymlinksInPath];
            
            // Load from file URL (most reliable method)
            NSURL* metallibURL = [NSURL fileURLWithPath:metallibPath];
            if (!metallibURL) {
                throw std::runtime_error(
                    std::string("Failed to create URL from path: ") +
                    [metallibPath UTF8String]
                );
            }
            
            library = [device newLibraryWithURL:metallibURL error:&error];
            
            if (!library) {
                NSString* errorDesc = error ? [error localizedDescription] : @"Unknown error";
                std::string errorMsg = "Failed to load Metal library from " + 
                    std::string([metallibPath UTF8String]) + ": " + 
                    std::string([errorDesc UTF8String]);
                throw std::runtime_error(errorMsg);
            }
        } else {
            // Fallback: try to use default library (if kernel was compiled into app bundle)
            library = [device newDefaultLibrary];
            if (!library) {
                throw std::runtime_error(
                    "Failed to find add_one_metal.metallib file. "
                    "Please ensure the Metal library is compiled and available."
                );
            }
        }
        
        // Get kernel function
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"add_one_kernel_nd"];
        if (!kernelFunction) {
            throw std::runtime_error("Metal kernel function 'add_one_kernel_nd' not found");
        }
        
        // Create compute pipeline
        id<MTLComputePipelineState> pipelineState = 
            [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            NSString* errorDesc = error ? [error localizedDescription] : @"Unknown error";
            throw std::runtime_error(
                std::string("Failed to create compute pipeline: ") + 
                [errorDesc UTF8String]
            );
        }
        
        // Convert int64_t arrays to int32_t for Metal (Metal doesn't support int64_t in buffers)
        // Note: This assumes values fit in int32_t (which is true for most practical cases)
        std::vector<int32_t> x_strides_int32(ndim);
        std::vector<int32_t> y_strides_int32(ndim);
        std::vector<int32_t> shape_int32(ndim);
        for (int i = 0; i < ndim; ++i) {
            x_strides_int32[i] = static_cast<int32_t>(x_strides_view[i]);
            y_strides_int32[i] = static_cast<int32_t>(y_strides_view[i]);
            shape_int32[i] = static_cast<int32_t>(shape_view[i]);
        }
        
        // Create buffers for metadata (using int32_t)
        size_t metadataSize = ndim * sizeof(int32_t);
        id<MTLBuffer> xStridesBuffer = [device newBufferWithBytes:x_strides_int32.data()
                                                           length:metadataSize
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> yStridesBuffer = [device newBufferWithBytes:y_strides_int32.data()
                                                           length:metadataSize
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> shapeBuffer = [device newBufferWithBytes:shape_int32.data()
                                                        length:metadataSize
                                                       options:MTLResourceStorageModeShared];
        
        // Create buffers for scalar parameters
        int ndimScalar = ndim;
        int32_t totalElementsScalar = static_cast<int32_t>(total_elements);
        id<MTLBuffer> ndimBuffer = [device newBufferWithBytes:&ndimScalar
                                                       length:sizeof(int)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> totalElementsBuffer = [device newBufferWithBytes:&totalElementsScalar
                                                                length:sizeof(int32_t)
                                                               options:MTLResourceStorageModeShared];
        
        // x_data and y_data are MTLBuffer object pointers (not data pointers)
        // Convert from void* to id<MTLBuffer>
        id<MTLBuffer> xBuffer = (__bridge id<MTLBuffer>)(x_data);
        id<MTLBuffer> yBuffer = (__bridge id<MTLBuffer>)(y_data);
        
        // Validate buffers
        if (!xBuffer || !yBuffer) {
            throw std::runtime_error("Invalid Metal buffer pointers");
        }
        
        // Get byte offsets from DLTensor (for Metal arrays with non-zero offset)
        size_t x_offset = static_cast<size_t>(x.byte_offset());
        size_t y_offset = static_cast<size_t>(y.byte_offset());
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            throw std::runtime_error("Failed to create Metal command buffer");
        }
        
        // Create compute command encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        if (!computeEncoder) {
            throw std::runtime_error("Failed to create Metal compute command encoder");
        }
        
        // Set compute pipeline state
        [computeEncoder setComputePipelineState:pipelineState];
        
        // Set buffers with byte offsets (for Metal arrays with non-zero offset)
        // Buffer indices must match those defined in add_one_metal.metal
        [computeEncoder setBuffer:xBuffer offset:x_offset atIndex:0];
        [computeEncoder setBuffer:yBuffer offset:y_offset atIndex:1];
        [computeEncoder setBuffer:xStridesBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:yStridesBuffer offset:0 atIndex:3];
        [computeEncoder setBuffer:shapeBuffer offset:0 atIndex:4];
        [computeEncoder setBuffer:ndimBuffer offset:0 atIndex:5];
        [computeEncoder setBuffer:totalElementsBuffer offset:0 atIndex:6];
        
        // Calculate threadgroup size
        NSUInteger threadgroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
        NSUInteger totalElementsNSU = static_cast<NSUInteger>(total_elements);
        if (threadgroupSize > totalElementsNSU) {
            threadgroupSize = totalElementsNSU;
        }
        MTLSize threadgroupSizeStruct = MTLSizeMake(threadgroupSize, 1, 1);
        
        // Calculate grid size
        MTLSize gridSize = MTLSizeMake(totalElementsNSU, 1, 1);
        
        // Dispatch threads
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSizeStruct];
        
        // End encoding
        [computeEncoder endEncoding];
        
        // Commit command buffer
        [commandBuffer commit];
        
        // Wait for completion
        [commandBuffer waitUntilCompleted];
        
        // Check for errors
        if (commandBuffer.error) {
            NSError* cmdError = commandBuffer.error;
            NSString* errorDesc = [cmdError localizedDescription];
            throw std::runtime_error(
                std::string("Metal command buffer error: ") + 
                [errorDesc UTF8String]
            );
        }
    }
}

}  // namespace tvmffi_test_fixtures

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_metal, tvmffi_test_fixtures::AddOneMetal);

