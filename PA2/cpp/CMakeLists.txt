cmake_minimum_required(VERSION 3.12)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-fPIC")

project( main )

find_package(Qt5 REQUIRED
    Core
    PrintSupport
    Gui
    Widgets
)


find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable( main main.cpp WeightMat.cpp NodeVec.cpp NeuralNet.cpp)
#target_link_libraries( main Qt5::Core ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} stdc++fs)
