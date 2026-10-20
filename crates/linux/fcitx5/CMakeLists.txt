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
#
# The install puts the engine and rewriter daemons, the ONNX Runtime the engine
# loads (SETSUNA_ORT_LIBRARY) and the model (SETSUNA_MODEL_MANIFEST) into
# SETSUNA_MODULE_DIR, where they are found with no config.toml.
# crates/linux/package.sh builds the release archive (crates/linux/install.sh
# installs it) with these.
cmake_minimum_required(VERSION 3.19)
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
set(CARGO_OUT "${CARGO_TARGET_DIR}/${CARGO_PROFILE_DIR}")
set(SETSUNA_RUST_LIB "${CARGO_OUT}/libsetsuna_linux.a")

option(SETSUNA_BUILD_SETTINGS "Build and install the settings GUI (needs WebKitGTK 4.1)" OFF)
set(SETSUNA_CARGO_PACKAGES -p setsuna-linux -p setsuna-engine -p setsuna-rewrite)
set(SETSUNA_DAEMONS "${CARGO_OUT}/setsuna-engine" "${CARGO_OUT}/setsuna-rewrite")
if(SETSUNA_BUILD_SETTINGS)
    list(APPEND SETSUNA_CARGO_PACKAGES -p setsuna-settings)
endif()

# Always run: cargo knows best whether anything changed.
add_custom_target(setsuna_rust ALL
    COMMAND cargo build ${CARGO_PROFILE_FLAG} ${SETSUNA_CARGO_PACKAGES}
            --target-dir "${CARGO_TARGET_DIR}"
    WORKING_DIRECTORY "${SETSUNA_REPO}"
    BYPRODUCTS "${SETSUNA_RUST_LIB}" ${SETSUNA_DAEMONS}
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
# The mixed input model, used when config.toml sets no [frontend] mixed_model.
install(DIRECTORY ../../ime-core/data/mixed/ DESTINATION "${SETSUNA_MODULE_DIR}/mixed")

# --- the daemons, ONNX Runtime and the model ---------------------------------

install(PROGRAMS ${SETSUNA_DAEMONS} DESTINATION "${SETSUNA_MODULE_DIR}")
if(SETSUNA_BUILD_SETTINGS)
    install(PROGRAMS "${CARGO_OUT}/setsuna-settings" DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

set(SETSUNA_ORT_LIBRARY "${CARGO_OUT}/libonnxruntime.so" CACHE FILEPATH
    "ONNX Runtime library installed next to the engine (crates/engine/fetch-onnxruntime.sh)")
if(EXISTS "${SETSUNA_ORT_LIBRARY}")
    get_filename_component(ort_file "${SETSUNA_ORT_LIBRARY}" REALPATH)
    install(PROGRAMS "${ort_file}" DESTINATION "${SETSUNA_MODULE_DIR}" RENAME libonnxruntime.so)
else()
    message(WARNING "${SETSUNA_ORT_LIBRARY} does not exist: the engine needs ONNX Runtime "
        "next to it (crates/engine/fetch-onnxruntime.sh), SETSUNA_ORT_DYLIB or [engine] ort_dylib")
endif()

# The manifest and every file it names, at the same paths relative to it, so
# the installed manifest resolves unchanged; a path leaving the artifact
# directory is refused.
set(SETSUNA_MODEL_MANIFEST "" CACHE FILEPATH
    "Conversion model manifest (*.manifest.json) installed with its files; empty installs none")
if(SETSUNA_MODEL_MANIFEST)
    get_filename_component(manifest "${SETSUNA_MODEL_MANIFEST}" ABSOLUTE)
    if(NOT EXISTS "${manifest}")
        message(FATAL_ERROR "SETSUNA_MODEL_MANIFEST ${manifest} does not exist")
    endif()
    get_filename_component(manifest_dir "${manifest}" DIRECTORY)
    get_filename_component(manifest_name "${manifest}" NAME)
    file(READ "${manifest}" manifest_json)
    string(JSON model_files_0 GET "${manifest_json}" proposal path)
    string(JSON model_files_1 GET "${manifest_json}" vocab)
    set(model_files "${model_files_0}" "${model_files_1}")
    string(JSON external ERROR_VARIABLE no_external GET "${manifest_json}" proposal external_data)
    if(no_external STREQUAL "NOTFOUND")
        string(JSON external_type TYPE "${manifest_json}" proposal external_data)
        if(external_type STREQUAL "ARRAY")
            string(JSON n LENGTH "${manifest_json}" proposal external_data)
            math(EXPR last "${n} - 1")
            foreach(i RANGE ${last})
                string(JSON f GET "${manifest_json}" proposal external_data ${i})
                list(APPEND model_files "${f}")
            endforeach()
        else()
            list(APPEND model_files "${external}")
        endif()
    endif()
    set(model_dest "${SETSUNA_MODULE_DIR}/models/onnx")
    # The engine reads the only manifest there: another model's is removed.
    install(CODE "
        file(GLOB old_manifests \"\$ENV{DESTDIR}${model_dest}/*manifest.json\")
        foreach(m IN LISTS old_manifests)
            get_filename_component(n \"\${m}\" NAME)
            if(NOT n STREQUAL \"${manifest_name}\")
                message(STATUS \"Removing: \${m}\")
                file(REMOVE \"\${m}\")
            endif()
        endforeach()")
    install(FILES "${manifest}" DESTINATION "${model_dest}")
    foreach(f IN LISTS model_files)
        if(IS_ABSOLUTE "${f}" OR f MATCHES "(^|[/\\\\])\\.\\.([/\\\\]|$)")
            message(FATAL_ERROR "manifest path ${f} leaves the artifact directory")
        endif()
        if(NOT EXISTS "${manifest_dir}/${f}")
            message(FATAL_ERROR "artifact file not found: ${manifest_dir}/${f}")
        endif()
        get_filename_component(sub "${f}" DIRECTORY)
        install(FILES "${manifest_dir}/${f}" DESTINATION "${model_dest}/${sub}")
    endforeach()
endif()
