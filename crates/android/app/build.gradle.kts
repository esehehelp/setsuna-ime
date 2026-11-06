// The Setsuna input method for Android. The native libraries come from
// ../build.ps1, which builds the Rust half (libsetsuna_android.so) and the
// engine (libsetsuna_engine.so) into build/rust/jniLibs before Gradle runs.

import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application") version "8.13.2"
    id("org.jetbrains.kotlin.android") version "2.2.21"
}

android {
    namespace = "io.github.esehehelp.setsuna"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.esehehelp.setsuna"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.2.1"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // installable as it is built; a release for others is signed
            // with a key of its own
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets["main"].apply {
        jniLibs.srcDir(layout.buildDirectory.dir("rust/jniLibs"))
        // the mixed input model (assets/mixed), unpacked into the files
        // directory on the first start
        assets.srcDir("../../ime-core/data")
    }

    packaging {
        jniLibs {
            // extracted to the native library directory: the engine is run
            // from there (an APK's libraries are not files to execute)
            useLegacyPackaging = true
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // libonnxruntime.so, which the engine loads (Windows pins the same release)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.28.0")
}
