## Gemini has the deps wrong - but the general idea is here 

# Start with Triton's base environment (everything CUDA is already here)
FROM nvcr.io/nvidia/tritonserver:26.02-py3

# 1. Install standard system tools needed to fetch and build things
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libssl-dev \
    libexpat1-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. OPTIX DEPENDENCY: Opticks needs the NVIDIA OptiX headers.
# Since OptiX is a header-only library API backed by drivers, you download 
# the OptiX SDK version matching your Opticks requirement and copy it in.
# (Note: Usually requires logging into NVIDIA Developer to get the exact version tarball)
COPY NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64 /opt/optix/

# 3. GEANT4 DEPENDENCY: Opticks often interfaces with Geant4 for initial event layouts
# We download, build, and install it right into the image.
WORKDIR /tmp/geant4-build
RUN curl -L https://geant4-data.web.cern.ch/releases/geant4-v11.1.0.tar.gz | tar -xz \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/geant4 -DGEANT4_INSTALL_DATA=ON ../geant4-v11.1.0 \
    && make -j$(nproc) install \
    && rm -rf /tmp/geant4-build

# 4. OPTICKS DEPENDENCY: Clone and install the native Opticks software package
# Pass the paths of the items we defined above so its build scripts can bind them.
WORKDIR /opt/opticks_source
RUN git clone https://bitbucket.org/simoncblyth/opticks.git . \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/opticks \
             -DGeant4_DIR=/opt/geant4/lib/CMake/Geant4 \
             -DOptiX_INSTALL_DIR=/opt/optix \
             .. \
    && make -j$(nproc) install

# 5. ENVIRONMENT PATHS: Ensure the server knows where to find these custom libraries 
# when loading your custom backend shared object later.
ENV OPTICKS_HOME=/opt/opticks
ENV LD_LIBRARY_PATH=/opt/opticks/lib:/opt/geant4/lib:/opt/tritonserver/lib:${LD_LIBRARY_PATH}

# Set the workspace back to a neutral spot
WORKDIR /workspace

