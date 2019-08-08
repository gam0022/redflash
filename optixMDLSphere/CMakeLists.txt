#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
if( GLUT_FOUND AND OPENGL_FOUND AND mdl_wrapper_FOUND )

    include_directories( ${GLUT_INCLUDE_DIR} ${mdl_wrapper_INCLUDE_DIR} )
    add_definitions( -DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA )

    # See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
    OPTIX_add_sample_executable( optixMDLSphere
        optixMDLSphere.cpp
        mdl_ibl.cu
        camera.cu
        shared_structs.h

        # These files are common among multiple samples
        mdl_textures.cu
        sphere_texcoord_tangent.cu
        random.h

        # Callable programs require separate compilation flags
        OPTIONS -dc
        )

    target_link_libraries( optixMDLSphere
        mdl_wrapper
        )
        
    set_target_properties( optixMDLSphere PROPERTIES COMPILE_DEFINITIONS TARGET_NAME="optixMDLSphere" )

else()
    # GLUT, OpenGL or mdl_wrapper not found
    message( "Disabling optixMDLSphere, which requires glut and opengl and mdl_wrapper." )
endif()
