# TFLite interpreter JNI is referenced reflectively by the runtime.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
