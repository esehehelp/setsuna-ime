# The fcitx5 addon: a C++ shim (setsuna.cpp) around the setsuna-linux Rust
# static library, which cargo builds as part of this project.
#
#   cmake -B build -DCMAKE_INSTALL_PREFIX=/usr crates/linux/fcitx5
#   cmake --build build && sudo cmake --install build
#
# The engine and models are looked for in SETSUNA_MODULE_DIR last (after
# config.toml and the SETSUNA_* environment variables). For development,
# point it at the cargo target dir: -DSETSUNA_MODULE_DIR=<repo>/target/release
# finds target/release/setsuna-engine and <repo>/local/models/onnx.
cmake_minimum_required(VERSION 3.16)
project(fcitx5-setsuna VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
find_package(Fcitx5Core 5.1 REQUIRED)
include("${FCITX_INSTALL_CMAKECONFIG_DIR}/Fcitx5Utils/Fcitx5CompilerSettings.cmake")

set(SETSUNA_MODULE_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/setsuna-ime" CACHE PATH
    "Directory searched last for setsuna-engine and the models")

# --- the Rust half -----------------------------------------------------------

get_filename_component(SETSUNA_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CARGO_TARGET_DIR "${SETSUNA_REPO}/target" CACHE PATH "cargo target directory")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_PROFILE_FLAG "")
    set(CARGO_PROFILE_DIR "debug")
else()
    set(CARGO_PROFILE_FLAG "--release")
    set(CARGO_PROFILE_DIR "release")
endif()
set(SETSUNA_RUST_LIB "${CARGO_TARGET_DIR}/${CARGO_PROFILE_DIR}/libsetsuna_linux.a")

# Always run: cargo knows best whether anything changed.
add_custom_target(setsuna_rust ALL
    COMMAND cargo build ${CARGO_PROFILE_FLAG} -p setsuna-linux
            --target-dir "${CARGO_TARGET_DIR}"
    WORKING_DIRECTORY "${SETSUNA_REPO}"
    BYPRODUCTS "${SETSUNA_RUST_LIB}"
    USES_TERMINAL
    COMMENT "cargo build -p setsuna-linux")

add_library(setsuna_rust_lib STATIC IMPORTED)
set_target_properties(setsuna_rust_lib PROPERTIES IMPORTED_LOCATION "${SETSUNA_RUST_LIB}")
add_dependencies(setsuna_rust_lib setsuna_rust)

# --- the addon ---------------------------------------------------------------

add_library(setsuna MODULE setsuna.cpp)
set_target_properties(setsuna PROPERTIES PREFIX "")
target_include_directories(setsuna PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_definitions(setsuna PRIVATE SETSUNA_MODULE_DIR="${SETSUNA_MODULE_DIR}")
# `cargo rustc -- --print native-static-libs`
target_link_libraries(setsuna PRIVATE Fcitx5::Core setsuna_rust_lib
    gcc_s util rt pthread m dl)
# Keep the Rust runtime's symbols out of the module's exports, so they never
# clash with another Rust addon loaded into the same fcitx5.
target_link_options(setsuna PRIVATE "LINKER:--exclude-libs,ALL")

install(TARGETS setsuna DESTINATION "${FCITX_INSTALL_ADDONDIR}")
configure_file(setsuna-addon.conf.in "${CMAKE_CURRENT_BINARY_DIR}/setsuna-addon.conf" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/setsuna-addon.conf"
    RENAME setsuna.conf DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon")
install(FILES setsuna-im.conf
    RENAME setsuna.conf DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod")
# The candidate panel themes (pick them in fcitx5's Classic User Interface
# settings) and the tray icons. themes/generate.py draws them.
install(DIRECTORY themes/setsuna themes/setsuna-dark
    DESTINATION "${FCITX_INSTALL_PKGDATADIR}/themes")
install(FILES icons/setsuna.svg icons/setsuna-hiragana.svg icons/setsuna-direct.svg
    DESTINATION "${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps")
