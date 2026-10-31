package dev.callguard.dsp

/** Fixed-size circular buffer of mono float samples. */
class RingBuffer(private val capacity: Int) {

    private val data = FloatArray(capacity)
    private var writeIndex = 0
    private var filled = 0

    @Synchronized
    fun write(src: FloatArray) {
        if (src.isEmpty()) return
        var copied = 0
        while (copied < src.size) {
            val n = minOf(src.size - copied, capacity - writeIndex)
            System.arraycopy(src, copied, data, writeIndex, n)
            writeIndex = (writeIndex + n) % capacity
            copied += n
        }
        filled = minOf(capacity, filled + src.size)
    }

    /** Returns the most recent [n] samples, oldest first, or null if not enough data. */
    @Synchronized
    fun readLast(n: Int): FloatArray? {
        if (n <= 0 || n > capacity || filled < n) return null
        val out = FloatArray(n)
        var start = writeIndex - n
        if (start < 0) start += capacity
        val first = minOf(n, capacity - start)
        System.arraycopy(data, start, out, 0, first)
        if (first < n) System.arraycopy(data, 0, out, first, n - first)
        return out
    }

    @get:Synchronized
    val available: Int get() = filled

    @Synchronized
    fun clear() {
        writeIndex = 0
        filled = 0
        java.util.Arrays.fill(data, 0f)
    }
}
