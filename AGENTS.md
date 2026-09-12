These are general engineering defaults, not unconditional requirements. Apply them only when they fit the approved task scope and the actual constraints of this repository
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- Study how established products solve the problem before designing a solution. Adopt their proven patterns and conventions rather than inventing an approach from scratch.

code风格规范：fail-fast 策略。isaaclab相关code必须避免为了 “所谓的code健壮性” 来添加不必要的保护性操作/fallback强行让仿真/训练运行下去。我需要将code问题在运行/训练中暴露出来。 同时审计/review时必须合理规划，不能反复review，过度审计，严格控制编译/diff/路径边界检查次数，减少过度串行的 fixture 修复、sandbox loopback、重复等待和过保守检查。你必须先证明操作路径，先把功能实现出来，等我确认没问题，然后才能添加护栏、变异/回归/遗留兼容性保护，或测试。或者只有等到我提起某个功能在什么情况下出现了问题之后再去补充相关的测试。要专注在功能实现本身上，而不是过度关注安全、护栏和各种测试
开始任何实现、调试、review 或文档更新前，必须先合理使用项目内 file-based memory system。（如果当前项目没有实现Memory机制请忽略）
注意：
1. 我们不是一个安全攻防项目，你有权力进行校验，但是禁止禁止禁止过度防御
2. 禁止写哈希和SHA256
3. 禁止反复的基本不可能出现的case写防御
4. 需要rubric的地方不要过度机械化
5. 任何等待任务直接sleep 30s 200s 600s  1800s或者更长时间（20h）来长时间等待，不要反复轮询。或者main agent派发worker等待，并行进行orchestratior编写等任务。
6. 接上一条，长等待任务（超过30min）为了不被干扰，将其运行在一个独立的tmux session中。
7. 调用工具的时候 我建议你promise.all来批量调取节省token
8. 每当你上下文被压缩进行一轮总结重新开始时 很多之前的我的引导信息命令都会重新输入你的上下文一次 这个时候不要去重复的回应过往的引导信息和提问等——你实际已经回复过了。 保持清晰的思维跟紧最新进度。

实机启动示例：
(sim) xinc@ai-precog-machine13:/mnt/nvme1/xinc/piper_sdk$ OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 4-7,14-17 python -m piper_sdk.deployment.manipulation   --checkpoint_path /mnt/nvme1/xinc/RoboDuet-IsaacLab/logs/rsl_rl/manipulation_direct/2026-08-24_11-57-39/model_5999.pt   --policy_steps 500   --run_policy   --target_pos_b 0.5 -0.2 0.4