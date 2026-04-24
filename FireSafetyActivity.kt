/*
 * 消防问答 Android 推理示例
 * 使用 llama.cpp Android JNI 接口加载 GGUF 模型
 *
 * 依赖 (build.gradle):
 *   implementation("com.github.ggerganov:llama.cpp:master")   // 或引入本地 .aar
 *
 * 权限 (AndroidManifest.xml):
 *   <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
 *
 * 模型文件放置: src/main/assets/fire_safety_q4_k_m.gguf
 * 或运行时从网络/SD卡加载。
 */

package com.example.firesafety

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

// ─────────────────────────────────────────────────────────
// 1. llama.cpp JNI 绑定 (需要将 libllama.so 打包进 APK)
// ─────────────────────────────────────────────────────────
object LlamaApi {
    init {
        System.loadLibrary("llama")  // 加载 libllama.so
    }

    // JNI 函数声明 — 对应 llama.cpp Android 封装层
    external fun llamaInit(modelPath: String, nCtx: Int, nThreads: Int): Long
    external fun llamaGenerate(ctxPtr: Long, prompt: String, maxTokens: Int,
                               temp: Float, topP: Float, repPenalty: Float): String
    external fun llamaFree(ctxPtr: Long)
    external fun llamaSystemInfo(): String
}

// ─────────────────────────────────────────────────────────
// 2. 推理引擎封装
// ─────────────────────────────────────────────────────────
class FireSafetyEngine(private val context: Context) {

    companion object {
        private const val TAG = "FireSafetyEngine"
        private const val MODEL_ASSET = "fire_safety_q4_k_m.gguf"  // assets 中的模型文件名
        private const val N_CTX = 512       // 上下文长度, 与训练时 max_length 一致
        private const val N_THREADS = 4     // 手机端推荐 4 线程
        private const val MAX_NEW_TOKENS = 256
        private const val TEMPERATURE = 0.3f
        private const val TOP_P = 0.9f
        private const val REP_PENALTY = 1.1f

        private val SYSTEM_PROMPT = """
            你是一个专业的消防安全助手，熟悉中国消防法律法规、
            消防安全技术规范及火灾预防知识。请根据相关法规条款给出准确、
            权威的回答。回答应简洁、专业，必要时注明依据的法规条款。
        """.trimIndent()
    }

    private var ctxPtr: Long = 0L
    private var isInitialized = false

    /** 将 assets 中的模型文件复制到 files 目录并初始化引擎 */
    suspend fun initialize(): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val modelFile = copyAssetToFiles(MODEL_ASSET)
            Log.i(TAG, "初始化模型: ${modelFile.absolutePath} (${modelFile.length() / 1024 / 1024} MB)")

            ctxPtr = LlamaApi.llamaInit(modelFile.absolutePath, N_CTX, N_THREADS)
            if (ctxPtr == 0L) error("模型加载失败: llamaInit 返回 null")

            isInitialized = true
            Log.i(TAG, "模型初始化成功 | ${LlamaApi.llamaSystemInfo()}")
        }
    }

    /** 构建 ChatML 格式 prompt */
    private fun buildPrompt(question: String): String = buildString {
        append("<|im_start|>system\n")
        append(SYSTEM_PROMPT)
        append("<|im_end|>\n")
        append("<|im_start|>user\n")
        append(question)
        append("<|im_end|>\n")
        append("<|im_start|>assistant\n")
    }

    /** 同步推理 (在 IO 线程调用) */
    suspend fun answer(question: String): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            check(isInitialized) { "引擎未初始化" }
            val prompt = buildPrompt(question)
            LlamaApi.llamaGenerate(
                ctxPtr, prompt, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REP_PENALTY
            ).trim()
        }
    }

    fun release() {
        if (ctxPtr != 0L) {
            LlamaApi.llamaFree(ctxPtr)
            ctxPtr = 0L
            isInitialized = false
        }
    }

    private fun copyAssetToFiles(assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        if (!outFile.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output, bufferSize = 8 * 1024 * 1024)
                }
            }
        }
        return outFile
    }
}

// ─────────────────────────────────────────────────────────
// 3. ViewModel
// ─────────────────────────────────────────────────────────
data class ChatMessage(val text: String, val isUser: Boolean)

sealed class UiState {
    object Initializing : UiState()
    object Ready        : UiState()
    object Loading      : UiState()
    data class Error(val msg: String) : UiState()
}

class FireSafetyViewModel(private val engine: FireSafetyEngine) : ViewModel() {

    private val _uiState = MutableStateFlow<UiState>(UiState.Initializing)
    val uiState: StateFlow<UiState> = _uiState

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    init {
        viewModelScope.launch {
            engine.initialize()
                .onSuccess { _uiState.value = UiState.Ready }
                .onFailure { _uiState.value = UiState.Error(it.message ?: "未知错误") }
        }
    }

    fun sendQuestion(question: String) {
        if (_uiState.value != UiState.Ready) return
        _uiState.value = UiState.Loading
        _messages.value = _messages.value + ChatMessage(question, isUser = true)

        viewModelScope.launch {
            engine.answer(question)
                .onSuccess { answer ->
                    _messages.value = _messages.value + ChatMessage(answer, isUser = false)
                    _uiState.value = UiState.Ready
                }
                .onFailure { err ->
                    _uiState.value = UiState.Error(err.message ?: "推理失败")
                }
        }
    }

    override fun onCleared() {
        super.onCleared()
        engine.release()
    }
}

// ─────────────────────────────────────────────────────────
// 4. Compose UI
// ─────────────────────────────────────────────────────────
class FireSafetyActivity : ComponentActivity() {

    private lateinit var engine: FireSafetyEngine
    private lateinit var viewModel: FireSafetyViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine    = FireSafetyEngine(applicationContext)
        viewModel = FireSafetyViewModel(engine)

        setContent {
            MaterialTheme {
                FireSafetyScreen(viewModel)
            }
        }
    }
}

@Composable
fun FireSafetyScreen(viewModel: FireSafetyViewModel) {
    val uiState  by viewModel.uiState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    var input    by remember { mutableStateOf("") }
    val scroll   = rememberScrollState()

    LaunchedEffect(messages.size) { scroll.animateScrollTo(scroll.maxValue) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            "消防安全问答助手",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        when (val s = uiState) {
            is UiState.Initializing -> LinearProgressIndicator(Modifier.fillMaxWidth())
            is UiState.Error -> Text("错误: ${s.msg}", color = MaterialTheme.colorScheme.error)
            else -> {}
        }

        // 聊天记录
        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(scroll)
        ) {
            messages.forEach { msg ->
                MessageBubble(msg)
                Spacer(Modifier.height(8.dp))
            }
            if (uiState is UiState.Loading) {
                CircularProgressIndicator(Modifier.padding(8.dp))
            }
        }

        Spacer(Modifier.height(8.dp))

        // 输入区
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                placeholder = { Text("输入消防问题...") },
                modifier = Modifier.weight(1f),
                maxLines = 3,
            )
            Spacer(Modifier.width(8.dp))
            Button(
                onClick = {
                    if (input.isNotBlank()) {
                        viewModel.sendQuestion(input.trim())
                        input = ""
                    }
                },
                enabled = uiState == UiState.Ready,
            ) {
                Text("发送")
            }
        }

        // 快捷问题
        Spacer(Modifier.height(8.dp))
        Text("常见问题:", style = MaterialTheme.typography.labelSmall)
        listOf(
            "单位消防安全职责有哪些？",
            "疏散通道宽度要求是多少？",
            "灭火器检查周期是多久？"
        ).forEach { q ->
            TextButton(
                onClick = { viewModel.sendQuestion(q) },
                enabled = uiState == UiState.Ready,
                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
            ) { Text(q, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

@Composable
fun MessageBubble(msg: ChatMessage) {
    val bgColor = if (msg.isUser)
        MaterialTheme.colorScheme.primaryContainer
    else
        MaterialTheme.colorScheme.surfaceVariant

    Surface(
        color = bgColor,
        shape = MaterialTheme.shapes.medium,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text(
            text = if (msg.isUser) "❓ ${msg.text}" else "✅ ${msg.text}",
            modifier = Modifier.padding(12.dp),
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}
