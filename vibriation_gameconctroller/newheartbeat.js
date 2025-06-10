// newheartbeat.js

document.addEventListener('DOMContentLoaded', () => {
    // 页面元素获取
    const bpmInput = document.getElementById('bpmInput');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const statusDiv = document.getElementById('status');
    
    // 全局变量
    let gamepad = null;
    let heartbeatTimeoutId = null;
    let currentStep = 0;
    let heartbeatPattern = [];

    // --- 【带包络的振动函数】 ---

    /**
     * 使用“起音-衰减”包络来播放一次平滑的振动，以减少机械回振。
     * @param {number} peakIntensity - 振动的峰值强度 (0.0 到 1.0)。
     * @param {number} weakPeakIntensity - 弱马达的峰值强度。
     * @param {number} totalDuration - 本次振动总时长 (毫秒)。
     * @param {function} onComplete - 振动序列播放完成后的回调函数。
     */
    function playVibrationWithEnvelope(peakIntensity, weakPeakIntensity, totalDuration, onComplete) {
        if (!gamepad || !gamepad.vibrationActuator || peakIntensity === 0) {
            if (onComplete) onComplete();
            return;
        }

        // 定义包络的时间分配比例：20%起音, 40%持续, 40%衰减
        const attackDuration = totalDuration * 0.2;
        const sustainDuration = totalDuration * 0.4;
        const decayDuration = totalDuration * 0.4;

        // 将每个阶段分解为2步，以获得更平滑的过渡
        const steps = [
            // 1. 起音 (Attack)
            { duration: attackDuration / 2, intensity: peakIntensity * 0.5 },
            { duration: attackDuration / 2, intensity: peakIntensity * 1.0 },
            // 2. 持续 (Sustain)
            { duration: sustainDuration, intensity: peakIntensity * 1.0 },
            // 3. 衰减 (Decay) - 这是消除回振的关键
            { duration: decayDuration / 2, intensity: peakIntensity * 0.5 },
            { duration: decayDuration / 2, intensity: 0.1 }
        ];

        let stepIndex = 0;

        function executeStep() {
            if (stepIndex >= steps.length) {
                if (onComplete) onComplete();
                return;
            }

            const current = steps[stepIndex];
            gamepad.vibrationActuator.playEffect('dual-rumble', {
                startDelay: 0,
                duration: current.duration,
                strongMagnitude: current.intensity,
                weakMagnitude: current.intensity * (weakPeakIntensity / peakIntensity) // 按比例缩放弱马达
            });
            
            stepIndex++;
            setTimeout(executeStep, current.duration);
        }

        executeStep();
    }
    
    // --- 【心跳主循环】 ---

    function heartbeatLoop() {
        if (!gamepad) {
            stopHeartbeat();
            return;
        }

        const step = heartbeatPattern[currentStep];

        // 驱动下一次调用的计时器
        heartbeatTimeoutId = setTimeout(heartbeatLoop, step.duration);

        // 如果是振动事件，则使用带包络的函数播放
        if (step.strongMagnitude > 0) {
            playVibrationWithEnvelope(step.strongMagnitude, step.weakMagnitude, step.duration, null);
        }
        
        currentStep = (currentStep + 1) % heartbeatPattern.length;
    }

    function startHeartbeat() {
        if (heartbeatTimeoutId) return;

        if (!gamepad || !gamepad.vibrationActuator) {
            statusDiv.textContent = '错误：未找到支持高级振动的手柄。';
            return;
        }

        const bpm = parseInt(bpmInput.value, 10) || 75;
        const S1_DURATION = 120, SYSTOLIC_PAUSE = 50, S2_DURATION = 70;
        const FIXED_DURATION = S1_DURATION + SYSTOLIC_PAUSE + S2_DURATION;
        const totalCycleDuration = 60000 / bpm;
        let diastolicPause = totalCycleDuration - FIXED_DURATION;

        if (diastolicPause < 50) {
            diastolicPause = 50;
            const maxBpm = Math.floor(60000 / (FIXED_DURATION + diastolicPause));
            statusDiv.textContent = `警告：心率过高！已限制在 ${maxBpm} BPM。`;
        } else {
            statusDiv.textContent = `模拟开始... 心率: ${bpm} bpm (模式: 平滑包络)`;
        }

        // 动态生成振动模式
        heartbeatPattern = [
            // 第一心音 (S1 - "咚") - 注意这里的duration现在是整个事件的总时长
            { duration: S1_DURATION, strongMagnitude: 1.0, weakMagnitude: 0.8 },
            // 短暂停
            { duration: SYSTOLIC_PAUSE, strongMagnitude: 0.0, weakMagnitude: 0.0 },
            // 第二心音 (S2 - "嗒")
            { duration: S2_DURATION, strongMagnitude: 0.7, weakMagnitude: 0.5 },
            // 长暂停
            { duration: Math.round(diastolicPause), strongMagnitude: 0.0, weakMagnitude: 0.0 }
        ];

        currentStep = 0;
        heartbeatLoop();
    }

    function stopHeartbeat() {
        clearTimeout(heartbeatTimeoutId);
        heartbeatTimeoutId = null;
        if (gamepad && gamepad.vibrationActuator) {
            gamepad.vibrationActuator.reset();
        }
        statusDiv.textContent = '模拟已停止。';
    }

    // --- 手柄连接逻辑 ---
    function scanGamepads() {
        const gamepads = navigator.getGamepads();
        for (const gp of gamepads) {
            if (gp && gp.vibrationActuator && gp.vibrationActuator.type === 'dual-rumble') {
                gamepad = gp;
                statusDiv.textContent = '已连接到兼容的手柄！可以开始模拟了。';
                return;
            }
        }
    }

    // 事件监听
    window.addEventListener('gamepadconnected', (e) => { gamepad = e.gamepad; /*...*/ });
    startButton.addEventListener('click', startHeartbeat);
    stopButton.addEventListener('click', stopHeartbeat);

    scanGamepads();
    if (!gamepad) {
        statusDiv.textContent = '请连接游戏手柄，并按任意键激活...';
    }
});
