# Install script for directory: C:/Users/gam0022/Dropbox/redflash

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/OptiX-Samples")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/gam0022/Dropbox/redflash/build/optixBuffersOfBuffers/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixCallablePrograms/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixCompressedTexture/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixConsole/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixDenoiser/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixDeviceQuery/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixDynamicGeometry/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixGeometryTriangles/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixHello/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixInstancing/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixMDLDisplacement/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixMDLExpressions/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixMDLSphere/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixMeshViewer/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixMotionBlur/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixParticles/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixPathTracer/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixPrimitiveIndexOffsets/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixRaycasting/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixSelector/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixSphere/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixSpherePP/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixSSIMPredictor/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixTextureSampler/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixTutorial/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixWhitted/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/optixRaymarching/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/primeInstancing/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/primeMasking/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/primeMultiBuffering/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/primeMultiGpu/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/primeSimple/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/primeSimplePP/cmake_install.cmake")
  include("C:/Users/gam0022/Dropbox/redflash/build/sutil/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/gam0022/Dropbox/redflash/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
