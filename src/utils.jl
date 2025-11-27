#=
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
=#

"""
    _get_root_array(arr) -> AbstractArray

Unwrap array wrappers (SubArray, ReshapedArray, etc.) to find the root array.
Uses Julia's `parent()` interface recursively.
"""
function _get_root_array(arr)
    current = arr

    # Keep unwrapping while parent() gives us something different
    while hasmethod(parent, Tuple{typeof(current)})
        next = parent(current)
        # Stop if parent() returns the same object (we've reached the bottom)
        if next === current
            break
        end
        current = next
    end

    return current
end

