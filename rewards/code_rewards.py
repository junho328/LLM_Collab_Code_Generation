import re
import signal
from typing import List, Tuple, Optional, Callable
import builtins

# Verbose toggle (can be set by training scripts)
VERBOSE = True

# Global step counter for logging (set by training scripts)
_GLOBAL_STEP = 0
_CURRENT_PHASE = "train"
_CURRENT_TURN = 1


def set_logging_context(step: int = 0, phase: str = "train", turn: int = 1):
    """Set the current logging context (called by training scripts)."""
    global _GLOBAL_STEP, _CURRENT_PHASE, _CURRENT_TURN
    _GLOBAL_STEP = step
    _CURRENT_PHASE = phase
    _CURRENT_TURN = turn

from rewards.code_utils import (
    TimeoutException,
    check_aux_call_without_assignment,
    check_aux_function_usage,
    check_function_definition,
    check_syntax,
    cleanup_code,
    concatenate_functions,
    extract_imports_from_prompt,
    extract_imports_from_code_prompt,
    extract_specific_function,
    extract_test_cases,
    is_wrapper_function,
    run_bigcodebench_tests,
    timeout_handler,
)


def execution_reward_aux(
    completion1: List[str],
    completion2: List[str],
    test_cases: List[str],
    entry_points: List[str],
    prompts: List[str] = None,  # Add prompts parameter
) -> List[float]:
    """
    Reward function for aux + main function collaboration on code tasks:

    LEVEL 1:
    - +0.4 reward if aux function is properly defined with return statement in completion1
    - +0.6 reward if main function (entry_point) is properly defined with return statement in completion2

    LEVEL 2:
    - +0.5 reward if concatenated code has no syntax errors

    LEVEL 3:
    - +0 to +1.0 reward proportional to correct assertions passed from check(candidate) tests
    - +0.5 bonus if at least one test passes AND main function uses aux function
    - +1.0 bonus if main function is NOT just a wrapper around aux function
    - -0.5 deduction if aux function is called but return value is ignored

    Maximum reward: 4.0 (updated from 3.5)
    """
    # Local print override based on VERBOSE
    if not VERBOSE:
        def print(*args, **kwargs):  # type: ignore
            return None
    else:
        print = builtins.print  # type: ignore

    rewards = []
    TEST_TIMEOUT = 10  # Timeout per individual test

    # Handle case where prompts is not provided
    if prompts is None:
        prompts = [""] * len(completion1)

    for c1, c2, test_code, entry_point, prompt in zip(
        completion1, completion2, test_cases, entry_points, prompts
    ):
        reward = 0.0

        print("\n" + "=" * 60)
        print("TESTING HUMANEVAL AUX + MAIN FUNCTION COLLABORATION")
        print("=" * 60)
        print(f"Entry point: {entry_point}")
        print(f"Maximum possible reward: 4.0 (Level 3 max: 3.0)")

        # Extract imports from prompt
        imports = extract_imports_from_prompt(prompt)
        if imports:
            print(f"\n--- EXTRACTED IMPORTS ---")
            print(imports)

        # Print raw completions for debugging
        print(f"\n--- RAW COMPLETION 1 (AUX) ---")
        print(repr(c1))
        print(f"\n--- RAW COMPLETION 2 (MAIN) ---")
        print(repr(c2))

        # Clean completions
        c1_clean = cleanup_code(c1)
        c2_clean = cleanup_code(c2)

        print(f"\n--- CLEANED COMPLETION 1 (AUX) ---")
        print(repr(c1_clean))
        print(f"\n--- CLEANED COMPLETION 2 (MAIN) ---")
        print(repr(c2_clean))

        # Extract specific functions for validation
        aux_func = extract_specific_function(c1, "aux")
        main_func = extract_specific_function(c2, entry_point)

        print(f"\n--- EXTRACTED AUX FUNCTION ---")
        print(repr(aux_func))
        print(f"\n--- EXTRACTED MAIN FUNCTION ---")
        print(repr(main_func))

        # ================================================================
        # LEVEL 1: FUNCTION DEFINITION REQUIREMENTS
        # ================================================================
        print("\n📋 LEVEL 1: FUNCTION DEFINITION REQUIREMENTS")
        print("-" * 50)

        level1_passed = True

        # 1.1 Check aux function in completion1 (+0.4)
        # Only give reward if aux is actually defined, but don't fail if it's empty
        aux_check_passed, aux_message = check_function_definition(
            c1, "aux", "Aux function"
        )

        if aux_check_passed:
            reward += 0.4
            print(f"✅ {aux_message}: +0.4 (total: {reward})")
        else:
            print(f"⚠️  {aux_message} (continuing without aux reward)")
            # Don't set level1_passed = False for aux - it's optional

        # 1.2 Check main function in completion2 (+0.6)
        main_check_passed, main_message = check_function_definition(
            c2, entry_point, f"Main function ({entry_point})"
        )

        if main_check_passed:
            reward += 0.6
            print(f"✅ {main_message}: +0.6 (total: {reward})")
        else:
            print(f"❌ {main_message}")
            level1_passed = False

        print(f"📊 Level 1: {'PASSED' if level1_passed else 'FAILED'}")

        if not level1_passed:
            print("⏹️  STOPPING: Function definition requirements not met")
            print(f"Final reward: {reward}")
            # Log completion to file
            try:
                from loggers.completion_logger import log_completion
                log_completion(
                    entry_point=entry_point,
                    aux_completion=c1,
                    main_completion=c2,
                    aux_cleaned=c1_clean,
                    main_cleaned=c2_clean,
                    aux_extracted=aux_func or "",
                    main_extracted=main_func or "",
                    reward=reward,
                    level1_reward=0.4 if aux_check_passed else 0.0,
                    phase=_CURRENT_PHASE,
                    turn=_CURRENT_TURN,
                    step=_GLOBAL_STEP,
                )
            except Exception:
                pass
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 2: SYNTAX REQUIREMENTS
        # ================================================================
        print("\n⚙️  LEVEL 2: SYNTAX REQUIREMENTS")
        print("-" * 40)

        # 2.1 Concatenate functions with imports
        # Use cleaned code (c1_clean, c2_clean) instead of extracted functions
        # This preserves class definitions that functions may depend on
        combined_code = concatenate_functions(c1_clean, c2_clean, imports)

        print("\n--- Combined Code ---")
        print(combined_code)
        print("--- End Code ---")

        # 2.2 Check combined syntax (+0.5)
        syntax_passed, syntax_message = check_syntax(combined_code, "Combined code")

        if syntax_passed:
            reward += 0.5
            print(f"✅ {syntax_message}: +0.5 (total: {reward})")
        else:
            print(f"❌ {syntax_message}")
            print("⏹️  STOPPING: Syntax requirements not met")
            print(f"Final reward: {reward}")
            # Log completion to file
            try:
                from loggers.completion_logger import log_completion
                log_completion(
                    entry_point=entry_point,
                    aux_completion=c1,
                    main_completion=c2,
                    aux_cleaned=c1_clean,
                    main_cleaned=c2_clean,
                    aux_extracted=aux_func or "",
                    main_extracted=main_func or "",
                    reward=reward,
                    level1_reward=0.4 if aux_check_passed else 0.0,
                    level2_reward=0.0,
                    phase=_CURRENT_PHASE,
                    turn=_CURRENT_TURN,
                    step=_GLOBAL_STEP,
                )
            except Exception:
                pass
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 3: TEST EXECUTION REQUIREMENTS
        # ================================================================
        print("\n🧪 LEVEL 3: TEST EXECUTION REQUIREMENTS")
        print("-" * 40)

        # Extract test cases
        test_cases_list = extract_test_cases(test_code, entry_point)
        if not test_cases_list:
            print("❌ No test cases found")
            # Log completion to file
            try:
                from loggers.completion_logger import log_completion
                log_completion(
                    entry_point=entry_point,
                    aux_completion=c1,
                    main_completion=c2,
                    aux_cleaned=c1_clean,
                    main_cleaned=c2_clean,
                    aux_extracted=aux_func or "",
                    main_extracted=main_func or "",
                    reward=reward,
                    level1_reward=0.4 if aux_check_passed else 0.0,
                    level2_reward=0.5,
                    phase=_CURRENT_PHASE,
                    turn=_CURRENT_TURN,
                    step=_GLOBAL_STEP,
                )
            except Exception:
                pass
            rewards.append(reward)
            continue

        print(f"📝 Found {len(test_cases_list)} test case(s)")

        # Initialize test tracking variables
        passed_tests = 0
        total_tests = len(test_cases_list)

        # 3.1 Execute tests (+0 to +1.0)
        timeout_count = 0  # Track number of timeouts
        MAX_TIMEOUTS = 3  # Stop testing after 3 timeouts

        try:
            # Create execution environment (no timeout needed for function definitions)
            exec_globals = {}
            exec(combined_code, exec_globals)
            print("✅ Code definitions loaded successfully")

            # Run individual test cases WITH INDIVIDUAL TIMEOUTS
            for i, test_case in enumerate(test_cases_list):
                # Check if we should stop testing due to too many timeouts
                if timeout_count >= MAX_TIMEOUTS:
                    remaining_tests = total_tests - i
                    print(
                        f"🛑 STOPPING TEST EXECUTION: {timeout_count} timeouts reached (limit: {MAX_TIMEOUTS})"
                    )
                    print(f"⏭️  Skipping remaining {remaining_tests} tests")
                    print("⚠️  No bonuses will be awarded due to excessive timeouts")
                    break

                try:
                    # SET TIMEOUT FOR EACH TEST INDIVIDUALLY
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TEST_TIMEOUT)

                    print(f"🧪 Running Test {i + 1}: {test_case}")

                    # Parse the test case to extract the function call and expected result
                    test_match = re.search(
                        r"assert\s+(\w+)\(([^)]*)\)\s*==\s*(.+)", test_case
                    )
                    if test_match:
                        func_name = test_match.group(1)
                        func_args = test_match.group(2)
                        expected_result = test_match.group(3)

                        # Execute the function call to get actual result
                        func_call = f"{func_name}({func_args})"
                        actual_result = eval(func_call, exec_globals)
                        expected_result_eval = eval(expected_result, exec_globals)

                        print(f"   📞 Function call: {func_call}")
                        print(f"   🎯 Expected: {expected_result_eval}")
                        print(f"   📤 Actual: {actual_result}")

                        # Execute the actual test
                        exec(test_case, exec_globals)
                        passed_tests += 1
                        print(f"✅ Test {i + 1}: PASSED")
                    else:
                        # Fallback if parsing fails
                        exec(test_case, exec_globals)
                        passed_tests += 1
                        print(f"✅ Test {i + 1}: PASSED")

                    # CLEAR TIMEOUT AFTER SUCCESSFUL TEST
                    signal.alarm(0)

                except TimeoutException:
                    signal.alarm(0)  # Clear timeout
                    timeout_count += 1
                    print(
                        f"⏰ Test {i + 1}: TIMEOUT after {TEST_TIMEOUT} seconds (timeout #{timeout_count})"
                    )
                    print("⚠️  Likely infinite recursion or infinite loop in function")

                except AssertionError as e:
                    signal.alarm(0)  # Clear timeout
                    # Try to show more details about the assertion failure
                    if (
                        "test_match" in locals()
                        and test_match
                        and "actual_result" in locals()
                        and "expected_result_eval" in locals()
                    ):
                        print(f"❌ Test {i + 1}: FAILED")
                        print(f"   🎯 Expected: {expected_result_eval}")
                        print(f"   📤 Actual: {actual_result}")
                        print(
                            f"   💥 Assertion failed: {actual_result} != {expected_result_eval}"
                        )
                    else:
                        print(f"❌ Test {i + 1}: FAILED (AssertionError: {str(e)})")

                except Exception as e:
                    signal.alarm(0)  # Clear timeout
                    print(f"❌ Test {i + 1}: FAILED (Error: {str(e)})")
                    print(f"   🧪 Test case: {test_case}")

            # Calculate proportional reward for test cases (0 to +1.0)
            if total_tests > 0:
                test_reward = (passed_tests / total_tests) * 1.0
                reward += test_reward
                print(f"📊 Tests passed: {passed_tests}/{total_tests}")
                print(f"✅ Test reward: +{test_reward:.2f} (total: {reward})")

        except Exception as e:
            print(f"❌ Code loading failed: {str(e)}")
            signal.alarm(0)

        # ================================================================
        # LEVEL 3 BONUS: AUX FUNCTION USAGE AND ANTI-WRAPPER BONUSES
        # ================================================================
        print("\n🎁 LEVEL 3 BONUS: COLLABORATION AND COMPLEXITY CHECKS")
        print("-" * 55)

        # Check if main function uses aux function AND at least one test passed
        # Bonuses are still available even if we hit timeout limit (as long as some tests passed)
        if (
            passed_tests > 0 and aux_func
        ):  # Only check if we have aux function and passed tests
            main_uses_aux = check_aux_function_usage(main_func, "aux")

            if main_uses_aux:
                bonus_reward = 0.5
                reward += bonus_reward
                print(
                    f"✅ Main function uses aux function: +{bonus_reward} (total: {reward})"
                )

                # Additional bonus for non-wrapper behavior
                is_wrapper = is_wrapper_function(main_func, "aux")

                if not is_wrapper:
                    anti_wrapper_bonus = 1.0
                    reward += anti_wrapper_bonus
                    print(
                        f"✅ Main function is NOT a simple wrapper: +{anti_wrapper_bonus} (total: {reward})"
                    )
                    print(f"🎉 FULL COLLABORATION BONUS ACHIEVED!")
                else:
                    print(
                        "⚠️  Main function appears to be a simple wrapper (no anti-wrapper bonus)"
                    )
                    print(
                        "💡 Consider adding more logic to the main function beyond just calling aux()"
                    )

                # Check for aux calls without assignment (deduction)
                has_ignored_calls, ignored_calls = check_aux_call_without_assignment(
                    main_func, "aux"
                )

                if has_ignored_calls:
                    deduction = 0.5
                    reward -= deduction
                    print(
                        f"⚠️  Aux function called but return value ignored: -{deduction} (total: {reward})"
                    )
                    print("💡 Problematic aux calls found:")
                    for call in ignored_calls:
                        print(f"   📍 {call}")
                    print(
                        "💭 Consider assigning aux() result to a variable or using it in expressions"
                    )
                else:
                    print("✅ All aux function calls properly use return values")

            else:
                print("⚠️  Main function does not use aux function (no bonuses)")
        else:
            if passed_tests == 0:
                print("⚠️  No tests passed - no bonus eligibility")
            if not aux_func:
                print("⚠️  No aux function defined - no bonus eligibility")

        # Show impact of early stopping due to timeouts
        if timeout_count >= MAX_TIMEOUTS:
            skipped_tests = total_tests - (
                passed_tests + timeout_count + (i + 1 - passed_tests - timeout_count)
            )
            if skipped_tests > 0:
                potential_lost_reward = (skipped_tests / total_tests) * 1.0
                print(
                    f"⚠️  EARLY STOPPING: Skipped {skipped_tests} tests due to {timeout_count} timeouts"
                )
                print(
                    f"💰 Potential lost Level 3 test reward: ~{potential_lost_reward:.2f}"
                )
                print(
                    "💡 Bonuses still awarded since some tests passed before timeout limit"
                )

        print(f"\n🏆 FINAL REWARD: {reward} / 4.0")
        
        # Log completion to file if logger is initialized
        try:
            from loggers.completion_logger import log_completion
            log_completion(
                entry_point=entry_point,
                aux_completion=c1,
                main_completion=c2,
                aux_cleaned=c1_clean,
                main_cleaned=c2_clean,
                aux_extracted=aux_func or "",
                main_extracted=main_func or "",
                reward=reward,
                level1_reward=0.4 if aux_check_passed else 0.0,
                level2_reward=0.5 if syntax_passed else 0.0,
                level3_reward=reward - (0.4 if aux_check_passed else 0.0) - (0.5 if syntax_passed else 0.0) - (0.6 if main_check_passed else 0.0),
                passed_tests=passed_tests if 'passed_tests' in dir() else 0,
                total_tests=total_tests if 'total_tests' in dir() else 0,
                phase=_CURRENT_PHASE,
                turn=_CURRENT_TURN,
                step=_GLOBAL_STEP,
            )
        except Exception:
            pass  # Silently ignore if logger not available
        
        rewards.append(reward)

    return rewards


def execution_reward_bigcodebench(
    completion1: List[str],
    completion2: List[str],
    test_cases: List[str],
    entry_points: List[str],
    code_prompts: List[str] = None,
    instruct_prompts: List[str] = None,
) -> List[float]:
    """
    Reward function for aux + main function collaboration on BigCodeBench tasks.
    
    BigCodeBench uses unittest-based tests instead of simple assert statements.

    LEVEL 1:
    - +0.2 reward if aux function is properly defined with return statement in completion1
    - +0.4 reward if main function (entry_point) is properly defined with return statement in completion2

    LEVEL 2:
    - +0.4 reward if concatenated code has no syntax errors

    LEVEL 3:
    - +0 to +1.5 reward proportional to correct tests passed
    - +0.5 bonus if at least one test passes AND main function uses aux function
    - +1.0 bonus if main function is NOT just a wrapper around aux function
    - -0.5 deduction if aux function is called but return value is ignored

    Maximum reward: 4.0
    """
    # Local print override based on VERBOSE
    if not VERBOSE:
        def print(*args, **kwargs):
            return None
    else:
        print = builtins.print

    rewards = []
    TEST_TIMEOUT = 5  # Shorter timeout for training speed (sandboxed execution)

    # Handle case where code_prompts is not provided
    if code_prompts is None:
        code_prompts = [""] * len(completion1)
    
    # Handle case where instruct_prompts is not provided
    if instruct_prompts is None:
        instruct_prompts = [""] * len(completion1)

    for c1, c2, test_code, entry_point, code_prompt, instruct_prompt in zip(
        completion1, completion2, test_cases, entry_points, code_prompts, instruct_prompts
    ):
        reward = 0.0

        print("\n" + "=" * 60)
        print("TESTING BIGCODEBENCH AUX + MAIN FUNCTION COLLABORATION")
        print("=" * 60)
        print(f"Entry point: {entry_point}")
        print(f"Maximum possible reward: 4.0")
        
        # Print task description if available
        if instruct_prompt:
            print(f"\n--- TASK DESCRIPTION ---")
            print(instruct_prompt)

        # Extract imports from code_prompt (BigCodeBench provides imports in code_prompt)
        imports = extract_imports_from_code_prompt(code_prompt)
        if imports:
            print(f"\n--- EXTRACTED IMPORTS ---")
            print(imports)

        # Print raw completions for debugging
        print(f"\n--- RAW COMPLETION 1 (AUX) ---")
        print(repr(c1))
        print(f"\n--- RAW COMPLETION 2 (MAIN) ---")
        print(repr(c2))

        # Clean completions
        c1_clean = cleanup_code(c1)
        c2_clean = cleanup_code(c2)

        # Extract specific functions for validation
        aux_func = extract_specific_function(c1, "aux")
        main_func = extract_specific_function(c2, entry_point)

        print(f"\n--- EXTRACTED AUX FUNCTION ---")
        print(repr(aux_func))
        print(f"\n--- EXTRACTED MAIN FUNCTION ---")
        print(repr(main_func))

        # ================================================================
        # LEVEL 1: FUNCTION DEFINITION REQUIREMENTS
        # ================================================================
        print("\n📋 LEVEL 1: FUNCTION DEFINITION REQUIREMENTS")
        print("-" * 50)

        level1_passed = True

        # 1.1 Check aux function in completion1 (+0.4)
        aux_check_passed, aux_message = check_function_definition(
            c1, "aux", "Aux function"
        )

        if aux_check_passed:
            reward += 0.2
            print(f"✅ {aux_message}: +0.2 (total: {reward})")
        else:
            print(f"⚠️  {aux_message} (continuing without aux reward)")

        # 1.2 Check main function in completion2 (+0.6)
        main_check_passed, main_message = check_function_definition(
            c2, entry_point, f"Main function ({entry_point})"
        )

        if main_check_passed:
            reward += 0.4
            print(f"✅ {main_message}: +0.4 (total: {reward})")
        else:
            print(f"❌ {main_message}")
            level1_passed = False

        print(f"📊 Level 1: {'PASSED' if level1_passed else 'FAILED'}")

        if not level1_passed:
            print("⏹️  STOPPING: Function definition requirements not met")
            print(f"Final reward: {reward}")
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 2: SYNTAX REQUIREMENTS
        # ================================================================
        print("\n⚙️  LEVEL 2: SYNTAX REQUIREMENTS")
        print("-" * 40)

        # 2.1 Concatenate functions with imports
        # Use cleaned code (c1_clean, c2_clean) instead of extracted functions
        # This preserves class definitions that functions may depend on
        combined_code = concatenate_functions(c1_clean, c2_clean, imports)

        print("\n--- Combined Code (truncated) ---")
        print(combined_code)
        print("--- End Code ---")

        # 2.2 Check combined syntax (+0.4)
        syntax_passed, syntax_message = check_syntax(combined_code, "Combined code")

        if syntax_passed:
            reward += 0.4
            print(f"✅ {syntax_message}: +0.4 (total: {reward})")
        else:
            print(f"❌ {syntax_message}")
            print("⏹️  STOPPING: Syntax requirements not met")
            print(f"Final reward: {reward}")
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 3: TEST EXECUTION REQUIREMENTS (BigCodeBench unittest)
        # ================================================================
        print("\n🧪 LEVEL 3: TEST EXECUTION REQUIREMENTS (BigCodeBench)")
        print("-" * 40)

        # Run BigCodeBench unittest-based tests
        passed_tests, total_tests, error_msg = run_bigcodebench_tests(
            combined_code, test_code, entry_point, timeout=TEST_TIMEOUT
        )

        print(f"📝 Test results: {passed_tests}/{total_tests}")
        if error_msg:
            print(f"⚠️  Error: {error_msg}")

        # Calculate proportional reward for test cases (0 to +1.5)
        if total_tests > 0:
            test_reward = (passed_tests / total_tests) * 1.5
            reward += test_reward
            print(f"✅ Test reward: +{test_reward:.2f} (total: {reward})")

        # ================================================================
        # LEVEL 3 BONUS: AUX FUNCTION USAGE AND ANTI-WRAPPER BONUSES
        # ================================================================
        print("\n🎁 LEVEL 3 BONUS: COLLABORATION AND COMPLEXITY CHECKS")
        print("-" * 55)

        if passed_tests > 0 and aux_func:
            main_uses_aux = check_aux_function_usage(main_func, "aux")

            if main_uses_aux:
                bonus_reward = 0.5
                reward += bonus_reward
                print(f"✅ Main function uses aux function: +{bonus_reward} (total: {reward})")

                # Additional bonus for non-wrapper behavior
                is_wrapper = is_wrapper_function(main_func, "aux")

                if not is_wrapper:
                    anti_wrapper_bonus = 1.0
                    reward += anti_wrapper_bonus
                    print(f"✅ Main function is NOT a simple wrapper: +{anti_wrapper_bonus} (total: {reward})")
                    print(f"🎉 FULL COLLABORATION BONUS ACHIEVED!")
                else:
                    print("⚠️  Main function appears to be a simple wrapper (no anti-wrapper bonus)")

                # Check for aux calls without assignment (deduction)
                has_ignored_calls, ignored_calls = check_aux_call_without_assignment(
                    main_func, "aux"
                )

                if has_ignored_calls:
                    deduction = 0.5
                    reward -= deduction
                    print(f"⚠️  Aux function called but return value ignored: -{deduction} (total: {reward})")
                else:
                    print("✅ All aux function calls properly use return values")
            else:
                print("⚠️  Main function does not use aux function (no bonuses)")
        else:
            if passed_tests == 0:
                print("⚠️  No tests passed - no bonus eligibility")
            if not aux_func:
                print("⚠️  No aux function defined - no bonus eligibility")

        print(f"\n🏆 FINAL REWARD: {reward} / 4.0")
        rewards.append(reward)

    return rewards


# =============================================================================
# Differential Reward Computation for Mental Simulation Mode
# =============================================================================


def compute_agent2_differential_rewards(
    inference_code: str,
    main_code: str,
    agent1_actual_code: str,
    code_reward: float,
    lambda_t: float,
    code_threshold: float = 1.5,
    similarity_func: Optional[Callable[[str, str], float]] = None,
) -> Tuple[float, float]:
    """
    Compute differential rewards for Agent 2's inference and main function tokens.
    
    This function implements the gated reward mechanism for mental simulation:
    - Inference tokens receive: R_corr + (lambda_t * R_sim) if R_corr >= threshold, else R_corr
    - Main function tokens receive: R_corr only
    
    The gating ensures Agent 2 doesn't focus on mimicking Agent 1 at the expense
    of code quality. Only when the code passes quality threshold does the
    similarity reward matter.
    
    Args:
        inference_code: Agent 2's inference/prediction of Agent 1's implementation.
        main_code: Agent 2's generated main function.
        agent1_actual_code: Agent 1's actual helper function implementation.
        code_reward: The code correctness reward (R_corr) from test execution.
        lambda_t: Current weight for similarity reward (from curriculum schedule).
        code_threshold: Minimum code reward to enable similarity reward (default 1.5).
            Approximately 37.5% of max 4.0 reward, meaning at least basic
            function structure and some tests passing.
        similarity_func: Optional function to compute similarity. If None, uses
            GraphCodeBERT-based similarity from infer_similarity module.
    
    Returns:
        Tuple of (inference_reward, main_reward):
        - inference_reward: Reward for inference tokens (potentially includes R_sim)
        - main_reward: Reward for main function tokens (R_corr only)
    
    Example:
        >>> r_infer, r_main = compute_agent2_differential_rewards(
        ...     inference_code="def aux(x): return x * 2",
        ...     main_code="def task_func(y): return aux(y) + 1",
        ...     agent1_actual_code="def aux(n): return n * 2",
        ...     code_reward=2.5,
        ...     lambda_t=0.3,
        ...     code_threshold=1.5,
        ... )
        >>> # r_infer will include similarity bonus since code_reward >= threshold
        >>> # r_main will just be 2.5
    """
    # Main function tokens always get just the code reward
    main_reward = code_reward
    
    # Check if code reward meets threshold for similarity bonus
    if code_reward >= code_threshold:
        # Compute similarity reward
        if similarity_func is not None:
            r_sim = similarity_func(inference_code, agent1_actual_code)
        else:
            # Use default GraphCodeBERT similarity
            try:
                from rewards.infer_similarity import compute_inference_similarity
                r_sim = compute_inference_similarity(inference_code, agent1_actual_code)
            except ImportError:
                # Fallback if similarity module not available
                r_sim = 0.0
        
        # Inference tokens get code reward + weighted similarity reward
        inference_reward = code_reward + lambda_t * r_sim
    else:
        # Gated: similarity reward is 0 when code quality is below threshold
        inference_reward = code_reward
    
    return inference_reward, main_reward


def compute_batch_differential_rewards(
    inference_codes: List[str],
    main_codes: List[str],
    agent1_actual_codes: List[str],
    code_rewards: List[float],
    lambda_t: float,
    code_threshold: float = 1.5,
    similarity_func: Optional[Callable[[str, str], float]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Compute differential rewards for a batch of samples.
    
    This is more efficient than calling compute_agent2_differential_rewards
    multiple times as it can batch similarity computations.
    
    Args:
        inference_codes: List of Agent 2's inferences.
        main_codes: List of Agent 2's main functions.
        agent1_actual_codes: List of Agent 1's actual implementations.
        code_rewards: List of code correctness rewards.
        lambda_t: Current weight for similarity reward.
        code_threshold: Minimum code reward to enable similarity reward.
        similarity_func: Optional custom similarity function.
    
    Returns:
        Tuple of (inference_rewards, main_rewards):
        - inference_rewards: List of rewards for inference tokens
        - main_rewards: List of rewards for main function tokens
    """
    n = len(inference_codes)
    if not (n == len(main_codes) == len(agent1_actual_codes) == len(code_rewards)):
        raise ValueError("All input lists must have the same length")
    
    inference_rewards = []
    main_rewards = []
    
    # Identify which samples qualify for similarity bonus
    qualifying_indices = [i for i, r in enumerate(code_rewards) if r >= code_threshold]
    
    # Compute similarity rewards in batch for qualifying samples
    similarity_rewards = [0.0] * n
    
    if qualifying_indices and lambda_t > 0:
        if similarity_func is not None:
            # Use provided similarity function
            for i in qualifying_indices:
                similarity_rewards[i] = similarity_func(
                    inference_codes[i], agent1_actual_codes[i]
                )
        else:
            # Use batch similarity computation
            try:
                from rewards.infer_similarity import compute_batch_inference_similarity
                qualifying_inferences = [inference_codes[i] for i in qualifying_indices]
                qualifying_actuals = [agent1_actual_codes[i] for i in qualifying_indices]
                batch_similarities = compute_batch_inference_similarity(
                    qualifying_inferences, qualifying_actuals
                )
                for idx, i in enumerate(qualifying_indices):
                    similarity_rewards[i] = batch_similarities[idx]
            except ImportError:
                # Fallback to individual computation
                try:
                    from rewards.infer_similarity import compute_inference_similarity
                    for i in qualifying_indices:
                        similarity_rewards[i] = compute_inference_similarity(
                            inference_codes[i], agent1_actual_codes[i]
                        )
                except ImportError:
                    pass  # Leave similarity rewards as 0
    
    # Compute final rewards
    for i in range(n):
        main_rewards.append(code_rewards[i])
        if i in qualifying_indices:
            inference_rewards.append(code_rewards[i] + lambda_t * similarity_rewards[i])
        else:
            inference_rewards.append(code_rewards[i])
    
    return inference_rewards, main_rewards


def create_gated_reward_wrapper(
    base_reward_func: Callable,
    lambda_t_getter: Callable[[], float],
    code_threshold: float = 1.5,
    similarity_func: Optional[Callable[[str, str], float]] = None,
) -> Callable:
    """
    Create a reward wrapper that computes both base code rewards and
    differential rewards for mental simulation mode.
    
    This wrapper is designed to be used with the MAGRPOTrainer.
    
    Args:
        base_reward_func: The original reward function (e.g., execution_reward_aux).
        lambda_t_getter: A callable that returns the current lambda(t) value.
        code_threshold: Minimum code reward for similarity bonus.
        similarity_func: Optional custom similarity function.
    
    Returns:
        Wrapped reward function that returns both base rewards and differential info.
    """
    def wrapped_reward_func(
        *agent_completions,
        batch_items=None,
        inference_codes=None,
        **kwargs,
    ):
        # Compute base code rewards
        base_rewards = base_reward_func(*agent_completions, batch_items=batch_items, **kwargs)
        
        # If inference codes provided, compute differential rewards
        if inference_codes is not None and len(agent_completions) >= 2:
            lambda_t = lambda_t_getter()
            agent1_codes = agent_completions[0]  # Agent 1's actual implementations
            
            inference_rewards = []
            main_rewards = []
            
            for i, (r, infer, actual) in enumerate(zip(
                base_rewards, inference_codes, agent1_codes
            )):
                r_infer, r_main = compute_agent2_differential_rewards(
                    inference_code=infer,
                    main_code=agent_completions[-1][i] if len(agent_completions[-1]) > i else "",
                    agent1_actual_code=actual,
                    code_reward=r,
                    lambda_t=lambda_t,
                    code_threshold=code_threshold,
                    similarity_func=similarity_func,
                )
                inference_rewards.append(r_infer)
                main_rewards.append(r_main)
            
            return {
                "base_rewards": base_rewards,
                "inference_rewards": inference_rewards,
                "main_rewards": main_rewards,
            }
        
        return base_rewards
    
    return wrapped_reward_func
