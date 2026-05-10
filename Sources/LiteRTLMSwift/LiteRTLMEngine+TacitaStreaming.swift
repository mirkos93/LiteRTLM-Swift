// LiteRTLMEngine+TacitaStreaming.swift
//
// Tacita-side fork addition. Drop this file under
// `Sources/LiteRTLMSwift/` of your fork of mylovelycodes/LiteRTLM-Swift
// (branch: tacita/streaming-multimodal). It only references public
// types from the upstream wrapper (LiteRTLMEngine, LiteRTLMError) and
// the C API exported by `CLiteRTLM.framework` headers — no upstream
// modifications needed, so a rebase against new upstream releases is
// trivial.
//
// Adds three things the upstream wrapper omits:
//   1. `conversationSendStreaming(...)` — token-streaming multimodal
//      turn via `litert_lm_conversation_send_message_stream`. Returns
//      an `AsyncThrowingStream<String, Error>` so the call site reads
//      identically to upstream's `generateStreaming(...)`.
//   2. `extraContext: [String: Any]` parameter (encoded as the JSON
//      string the C API expects). The chat-template Jinja branch
//      `{% if enable_thinking %}<|think|>` only fires when this map
//      contains `"enable_thinking": true`, so reasoning on Gemma 4
//      depends entirely on this surface.
//   3. `cancelProcess()` — hard interrupt via
//      `litert_lm_conversation_cancel_process`, distinct from the
//      cooperative Swift `Task.cancel()` semantics. The runtime
//      acknowledges by emitting `isFinal=true` to the stream
//      callback so the executor unwinds cleanly.
//
// Plus: in-flight re-stitching of Gemma 4's structured channel
// markers — `<|channel>thought\n…\n<channel|>` becomes
// `<think>…</think>` so Tacita's shared Dart-side
// `ThinkingStreamParser` works identically across runtimes
// (fllama emits `<think>` natively; Android's Kotlin SDK has
// `Channel(...)` declarations; iOS does the substitution here).
//
// Thread / memory safety contract:
//   * The C callback fires on the inference queue's worker thread.
//     We forward into the AsyncThrowingStream continuation, which
//     is itself thread-safe.
//   * The opaque `callback_data` pointer carries a `passRetained`
//     boxed state, released exactly once when `isFinal` arrives or
//     when the C call returns nonzero.
//   * `cancelProcess()` is callable from any thread.

import Foundation
import CLiteRTLM

extension LiteRTLMEngine {

    // MARK: - Public streaming surface

    /// Streams a multimodal turn (text + optional images + optional
    /// audio) chunk-by-chunk. Mirrors upstream's
    /// `conversationSend(...)` parameter list and adds:
    ///
    /// - `extraContext`: forwarded to the C API as a JSON object,
    ///   the Jinja chat-template's source of truth for runtime
    ///   knobs (`enable_thinking`, `tool_choice`, …). Pass `[:]`
    ///   for the default behaviour.
    /// - `restitchThinkingMarkers`: when `true`, in-flight rewrite
    ///   `<|channel>thought\n…\n<channel|>` → `<think>…</think>`
    ///   on the streamed text before yielding. The substitution is
    ///   safe across chunk boundaries (we hold a tiny rolling
    ///   buffer for partial markers).
    public func conversationSendStreaming(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        extraContext: [String: Any] = [:],
        maxImageDimension: Int = 1024,
        restitchThinkingMarkers: Bool = true
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            inferenceQueue.async { [self] in
                do {
                    // NOTE: ensureReady() is `private` in the upstream wrapper
                    // and therefore inaccessible from this separate-file extension
                    // in Swift (only same-file `private` access is allowed within
                    // an extension). The equivalent safety contract is enforced by
                    // the `multimodalConversation` guard below: the bridge calls
                    // `openConversation()` during `loadModel` and only exposes
                    // the engine to callers after that succeeds, so `conversation`
                    // will be non-nil iff the engine is ready.
                    guard let conversation = multimodalConversation else {
                        throw LiteRTLMError.inferenceFailure(
                            "No persistent conversation open — call openConversation() first"
                        )
                    }

                    // Stage media to disk (the C API takes file paths).
                    var tempURLs: [URL] = []
                    var audioPaths: [String] = []
                    var imagePaths: [String] = []
                    for (i, data) in audioData.enumerated() {
                        guard !data.isEmpty else {
                            throw LiteRTLMError.inferenceFailure("Audio data \(i + 1) is empty")
                        }
                        let url = Self.makeTempURL(extension: audioFormat.rawValue)
                        try data.write(to: url)
                        tempURLs.append(url)
                        audioPaths.append(url.path)
                    }
                    for (i, data) in imagesData.enumerated() {
                        guard let jpegData = Self.prepareImageForVision(
                            data, maxDimension: maxImageDimension
                        ) else {
                            throw LiteRTLMError.inferenceFailure(
                                "Failed to convert image \(i + 1) to JPEG"
                            )
                        }
                        let url = Self.makeTempURL(extension: "jpg")
                        try jpegData.write(to: url)
                        tempURLs.append(url)
                        imagePaths.append(url.path)
                    }

                    let messageJSON = Self.buildMultimodalMessageJSON(
                        audioPaths: audioPaths,
                        imagePaths: imagePaths,
                        text: prompt
                    )
                    let extraContextJSON: String?
                    if extraContext.isEmpty {
                        extraContextJSON = nil
                    } else if
                        let data = try? JSONSerialization.data(
                            withJSONObject: extraContext
                        ),
                        let str = String(data: data, encoding: .utf8)
                    {
                        extraContextJSON = str
                    } else {
                        extraContextJSON = nil
                    }

                    let streamDone = DispatchSemaphore(value: 0)
                    let state = TacitaStreamState(
                        continuation: continuation,
                        doneSemaphore: streamDone,
                        restitchThinking: restitchThinkingMarkers,
                        tempURLs: tempURLs
                    )
                    let statePtr = Unmanaged.passRetained(state).toOpaque()

                    let callback: LiteRtLmStreamCallback = {
                        callbackData, chunk, isFinal, errorMsg in
                        guard let cbData = callbackData else { return }
                        let st = Unmanaged<TacitaStreamState>
                            .fromOpaque(cbData)
                            .takeUnretainedValue()

                        let errorMessage: String? = {
                            guard let errorMsg else { return nil }
                            let msg = String(cString: errorMsg)
                            return msg.isEmpty ? nil : msg
                        }()

                        if let chunk, errorMessage == nil {
                            let text = String(cString: chunk)
                            if !text.isEmpty {
                                st.deliverChunk(text, isFinal: isFinal)
                            }
                        }

                        if isFinal || errorMessage != nil {
                            // Flush any rolling buffer left over from
                            // the channel-marker re-stitcher before
                            // closing the stream.
                            st.flushRollingBuffer()
                            if let error = errorMessage {
                                st.continuation.finish(
                                    throwing: LiteRTLMError.inferenceFailure(error)
                                )
                            } else {
                                st.continuation.finish()
                            }
                            st.cleanupTempFiles()
                            let semaphore = st.doneSemaphore
                            Unmanaged<TacitaStreamState>
                                .fromOpaque(cbData)
                                .release()
                            semaphore.signal()
                        }
                    }

                    let result: Int32
                    if let ec = extraContextJSON {
                        result = messageJSON.withCString { msgPtr in
                            ec.withCString { ecPtr in
                                litert_lm_conversation_send_message_stream(
                                    conversation, msgPtr, ecPtr,
                                    callback, statePtr
                                )
                            }
                        }
                    } else {
                        result = messageJSON.withCString { msgPtr in
                            litert_lm_conversation_send_message_stream(
                                conversation, msgPtr, nil,
                                callback, statePtr
                            )
                        }
                    }

                    if result != 0 {
                        Unmanaged<TacitaStreamState>
                            .fromOpaque(statePtr)
                            .release()
                        for url in tempURLs {
                            try? FileManager.default.removeItem(at: url)
                        }
                        continuation.finish(throwing: LiteRTLMError.inferenceFailure(
                            "Failed to start multimodal stream (code \(result))"
                        ))
                        return
                    }

                    streamDone.wait()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Hard-interrupts the active multimodal conversation. Safe from
    /// any thread; the runtime emits `isFinal=true` to the stream
    /// callback shortly after, which unwinds the streaming task.
    /// No-op when no conversation is open.
    public func cancelProcess() {
        inferenceQueue.async { [self] in
            guard let conversation = multimodalConversation else { return }
            litert_lm_conversation_cancel_process(conversation)
        }
    }
}

// MARK: - Stream callback state (private, file-scope)

private final class TacitaStreamState {
    init(
        continuation: AsyncThrowingStream<String, Error>.Continuation,
        doneSemaphore: DispatchSemaphore,
        restitchThinking: Bool,
        tempURLs: [URL]
    ) {
        self.continuation = continuation
        self.doneSemaphore = doneSemaphore
        self.restitchThinking = restitchThinking
        self.tempURLs = tempURLs
    }

    let continuation: AsyncThrowingStream<String, Error>.Continuation
    let doneSemaphore: DispatchSemaphore
    let restitchThinking: Bool
    private let tempURLs: [URL]

    /// Holds a partial fragment that *might* contain the start of a
    /// channel marker so we don't accidentally yield half a token like
    /// `<|chann` and miss the rewrite. Flushed at end-of-stream.
    private var rolling = ""

    /// Open/close state for the running thinking block, so a single
    /// thinking text spread across many chunks emits exactly one
    /// `<think>` opener and `</think>` closer.
    private var thinkingOpen = false
    private var thinkingClosed = false

    func deliverChunk(_ text: String, isFinal: Bool) {
        guard restitchThinking else {
            continuation.yield(text)
            return
        }
        rolling += text
        let (out, remainder) = Self.transformAndSplit(
            rolling,
            thinkingOpen: &thinkingOpen,
            thinkingClosed: &thinkingClosed
        )
        rolling = remainder
        if !out.isEmpty { continuation.yield(out) }
    }

    func flushRollingBuffer() {
        guard restitchThinking, !rolling.isEmpty else {
            rolling = ""
            return
        }
        // Final flush: any unmatched marker is forwarded as-is.
        // We intentionally do NOT inject a closing `</think>` if a
        // thinking block was never sealed by the model — that would
        // hide a malformed stream.
        continuation.yield(rolling)
        rolling = ""
    }

    func cleanupTempFiles() {
        for url in tempURLs {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Walks `buffer` looking for Gemma 4 channel boundaries and
    /// rewrites them into `<think>...</think>` for downstream
    /// parsers. Returns `(safeToEmit, leftoverForNextCall)`.
    ///
    /// We keep the rewrite simple by only recognizing exactly two
    /// markers:
    ///     `<|channel>thought\n` → opens a thinking block
    ///     `<channel|>`         → closes it
    ///
    /// Anything between them is forwarded verbatim. Anything outside
    /// is forwarded verbatim. The ROLLING buffer is what protects us
    /// from yielding a chunk that ends in the middle of a marker.
    static func transformAndSplit(
        _ buffer: String,
        thinkingOpen: inout Bool,
        thinkingClosed: inout Bool
    ) -> (out: String, remainder: String) {
        // Heuristic: keep at most the longest possible unfinished
        // marker as the rolling tail (15 chars covers
        // "<|channel>thought" and "<channel|>" with margin).
        let maxMarker = 18
        var work = buffer
        var out = ""
        let openMarker = "<|channel>thought"
        let closeMarker = "<channel|>"
        while true {
            if let openRange = work.range(of: openMarker) {
                // Emit text before the marker untouched.
                out.append(contentsOf: work[..<openRange.lowerBound])
                if !thinkingOpen {
                    out.append("<think>")
                    thinkingOpen = true
                }
                // Drop the marker itself; if a leading newline
                // follows, also drop it so the reasoning panel
                // starts cleanly without the literal `\n`.
                var afterIdx = openRange.upperBound
                if afterIdx < work.endIndex, work[afterIdx] == "\n" {
                    afterIdx = work.index(after: afterIdx)
                }
                work = String(work[afterIdx...])
                continue
            }
            if let closeRange = work.range(of: closeMarker) {
                out.append(contentsOf: work[..<closeRange.lowerBound])
                if thinkingOpen && !thinkingClosed {
                    out.append("</think>")
                    thinkingClosed = true
                }
                work = String(work[closeRange.upperBound...])
                continue
            }
            break
        }
        // Hold back the tail that might be a partial marker.
        if work.count <= maxMarker {
            return (out, work)
        }
        let split = work.index(work.endIndex, offsetBy: -maxMarker)
        out.append(contentsOf: work[..<split])
        return (out, String(work[split...]))
    }
}
