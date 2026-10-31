package dev.callguard.audio

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import java.io.IOException
import java.nio.ByteOrder

/**
 * Decodes any container/codec the platform supports into 16 kHz mono float PCM.
 *
 * Used by the "analyse a recording" path. No network, no upload: the file is read
 * through the content resolver, decoded by MediaCodec, and fed straight to the
 * detector.
 */
object FileAudioDecoder {

    data class Decoded(
        val samples: FloatArray,
        val sourceRate: Int,
        val channels: Int,
    ) {
        val durationSeconds: Float
            get() = if (sourceRate > 0) samples.size.toFloat() / sourceRate else 0f
    }

    private const val TAG = "CallGuard/Decode"
    private const val TIMEOUT_US = 10_000L

    fun decode(
        context: Context,
        uri: Uri,
        maxSeconds: Float = 300f,
        onProgress: (Float) -> Unit = {},
    ): Decoded {
        val descriptor = context.contentResolver.openFileDescriptor(uri, "r")
            ?: throw IOException("Could not open the selected file")

        descriptor.use { pfd ->
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(pfd.fileDescriptor)

                var trackIndex = -1
                var format: MediaFormat? = null
                for (i in 0 until extractor.trackCount) {
                    val candidate = extractor.getTrackFormat(i)
                    val mime = candidate.getString(MediaFormat.KEY_MIME) ?: continue
                    if (mime.startsWith("audio/")) {
                        trackIndex = i
                        format = candidate
                        break
                    }
                }
                val trackFormat = format ?: throw IOException("The file has no audio track")
                if (trackIndex < 0) throw IOException("The file has no audio track")
                extractor.selectTrack(trackIndex)

                val mime = trackFormat.getString(MediaFormat.KEY_MIME)
                    ?: throw IOException("Unknown audio codec")
                val durationUs = if (trackFormat.containsKey(MediaFormat.KEY_DURATION)) {
                    trackFormat.getLong(MediaFormat.KEY_DURATION)
                } else {
                    0L
                }

                var sourceRate = trackFormat.intOr(MediaFormat.KEY_SAMPLE_RATE, 44100)
                var channels = trackFormat.intOr(MediaFormat.KEY_CHANNEL_COUNT, 1)
                var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT

                val sink = FloatSink()
                val codec = MediaCodec.createDecoderByType(mime)
                codec.configure(trackFormat, null, null, 0)
                codec.start()

                val info = MediaCodec.BufferInfo()
                var inputDone = false
                var outputDone = false
                var guard = 0

                try {
                    while (!outputDone && guard++ < 5_000_000) {
                        if (!inputDone) {
                            val inIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                            if (inIndex >= 0) {
                                val buffer = codec.getInputBuffer(inIndex)
                                if (buffer == null) {
                                    codec.queueInputBuffer(
                                        inIndex, 0, 0, 0,
                                        MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                                    )
                                    inputDone = true
                                } else {
                                    val sampleSize = extractor.readSampleData(buffer, 0)
                                    if (sampleSize < 0) {
                                        codec.queueInputBuffer(
                                            inIndex, 0, 0, 0,
                                            MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                                        )
                                        inputDone = true
                                    } else {
                                        codec.queueInputBuffer(
                                            inIndex, 0, sampleSize,
                                            extractor.sampleTime, 0,
                                        )
                                        extractor.advance()
                                    }
                                }
                            }
                        }

                        val outIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US)
                        when {
                            outIndex >= 0 -> {
                                val buffer = codec.getOutputBuffer(outIndex)
                                if (buffer != null && info.size > 0) {
                                    buffer.position(info.offset)
                                    buffer.limit(info.offset + info.size)
                                    if (pcmEncoding == AudioFormat.ENCODING_PCM_FLOAT) {
                                        val floats = FloatArray(info.size / 4)
                                        buffer.order(ByteOrder.nativeOrder())
                                            .asFloatBuffer()
                                            .get(floats)
                                        sink.addFloats(floats)
                                    } else {
                                        val shorts = ShortArray(info.size / 2)
                                        buffer.order(ByteOrder.nativeOrder())
                                            .asShortBuffer()
                                            .get(shorts)
                                        sink.addShorts(shorts)
                                    }
                                }
                                codec.releaseOutputBuffer(outIndex, false)

                                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                                    outputDone = true
                                }
                                if (durationUs > 0) {
                                    onProgress(
                                        (info.presentationTimeUs.toFloat() / durationUs)
                                            .coerceIn(0f, 1f),
                                    )
                                }
                                if (sink.size > maxSeconds * sourceRate * channels) {
                                    outputDone = true
                                }
                            }

                            outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                                val outputFormat = codec.outputFormat
                                sourceRate = outputFormat.intOr(MediaFormat.KEY_SAMPLE_RATE, sourceRate)
                                channels = outputFormat.intOr(MediaFormat.KEY_CHANNEL_COUNT, channels)
                                if (outputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                                    pcmEncoding = outputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                                }
                                Log.i(TAG, "decoder output ${sourceRate}Hz x$channels enc=$pcmEncoding")
                            }
                        }
                    }
                } finally {
                    runCatching { codec.stop() }
                    runCatching { codec.release() }
                }

                val mono = Resampler.toMono16k(sink.toArray(), sourceRate, channels)
                if (mono.isEmpty()) throw IOException("Decoded no audio samples")
                onProgress(1f)
                return Decoded(mono, sourceRate, channels)
            } finally {
                runCatching { extractor.release() }
            }
        }
    }

    private fun MediaFormat.intOr(key: String, fallback: Int): Int =
        if (containsKey(key)) getInteger(key) else fallback

    private class FloatSink(initial: Int = 1 shl 18) {
        private var data = FloatArray(initial)
        var size = 0
            private set

        fun addShorts(shorts: ShortArray) {
            ensure(shorts.size)
            for (s in shorts) data[size++] = s / 32768f
        }

        fun addFloats(floats: FloatArray) {
            ensure(floats.size)
            System.arraycopy(floats, 0, data, size, floats.size)
            size += floats.size
        }

        private fun ensure(extra: Int) {
            if (size + extra <= data.size) return
            var next = data.size
            while (next < size + extra) next = next shl 1
            data = data.copyOf(next)
        }

        fun toArray(): FloatArray = data.copyOf(size)
    }
}
