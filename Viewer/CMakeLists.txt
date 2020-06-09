find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_library(CurveViewer
    CurveViewer.cpp
    MainWindow.cpp
    RunningCurveViewer.cpp

    CurveViewer.h
    MainWindow.h
    RunningCurveViewer.h
)

target_include_directories(CurveViewer
    PUBLIC .
)

target_link_libraries(CurveViewer
    Twisty
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    ${OPENGL_gl_LIBRARY}
    glu32
)

# add_executable(PathPerturbViewer
#     PathPerturbViewer.cpp

#     PathPerturbViewerWidget.cpp
#     PathPerturbViewerWidget.h
# )

# target_link_libraries(PathPerturbViewer
#     CurveViewer
# )

add_executable(PathBatchViewer
    PathBatchViewer.cpp
    PathBatchViewerWidget.cpp

    PathBatchViewerWidget.h
)

target_link_libraries(PathBatchViewer
    CurveViewer
)