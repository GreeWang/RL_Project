SEEDS=(42 123 456 789 1024)
REPORT_SEEDS=(42 123 456)
BASE_ARGS="--episodes 800 --eval-every 20 --eval-episodes 10 --max-steps 100 --goal-reward 10.0"
REPORT_ARGS="--episodes 1000 --eval-every 20 --eval-episodes 10 --max-steps 100 --goal-reward 10.0 --alpha 0.1 --gamma 0.9 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.995"

echo "Experiments training begin"

# ==========================================
# 1. Baseline vs Random Policy
# ==========================================
echo "1/5 Baseline vs Random Policy"
for s in "${SEEDS[@]}"; do
    python train.py --exp-name baseline \
        --outdir results/baseline/seed_${s} \
        --seed ${s} ${BASE_ARGS}
done
for s in "${SEEDS[@]}"; do
    python train.py --exp-name random_policy \
        --outdir results/random_policy/seed_${s} \
        --seed ${s} \
        --alpha 1e-4 --epsilon-start 1.0 --epsilon-end 1.0 --epsilon-decay 1.0 ${BASE_ARGS}
done

# ==========================================
# 2. Epsilon Decay comparision (Fixed/Standard/Fast/Slow)
# ==========================================
echo "2/5 Epsilon Decay Comparison"
# Fixed ε=0.1
for s in "${SEEDS[@]}"; do
    python train.py --exp-name eps_fixed_0.1 \
        --outdir results/epsilon/fixed_0.1/seed_${s} \
        --seed ${s} \
        --epsilon-start 0.1 --epsilon-end 0.1 --epsilon-decay 1.0 ${BASE_ARGS}
done
# Standard Decay (1.0 → 0.05, decay=0.99)
for s in "${SEEDS[@]}"; do
    python train.py --exp-name eps_std \
        --outdir results/epsilon/standard/seed_${s} \
        --seed ${s} \
        --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.99 ${BASE_ARGS}
done
# Fast Decay
for s in "${SEEDS[@]}"; do
    python train.py --exp-name eps_fast \
        --outdir results/epsilon/fast/seed_${s} \
        --seed ${s} \
        --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.95 ${BASE_ARGS}
done
# Slow Decay
for s in "${SEEDS[@]}"; do
    python train.py --exp-name eps_slow \
        --outdir results/epsilon/slow/seed_${s} \
        --seed ${s} \
        --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.999 ${BASE_ARGS}
done

# ==========================================
# 3. Discount Factor γ comparision
# ==========================================
echo "3/5 Gamma (Discount Factor) Comparison"
for g in 0.5 0.8 0.95 0.99; do
    for s in "${SEEDS[@]}"; do
        python train.py --exp-name gamma_${g} \
            --outdir results/gamma_comparison/gamma_${g}/seed_${s} \
            --seed ${s} \
            --gamma ${g} ${BASE_ARGS}
    done
done

# ==========================================
# 4. Reward & Map Difficulty
# ==========================================
echo "4/5 Reward & Map Difficulty"
for r in sparse dense; do
    if [[ "$r" == "sparse" ]]; then
        step=0.0
        trap_pen=-10.0
    else
        step=-0.1
        trap_pen=-5.0
    fi
    for s in "${SEEDS[@]}"; do
        python train.py --exp-name reward_${r} \
            --outdir results/reward/${r}/seed_${s} \
            --seed ${s} \
            --step-penalty ${step} --trap-penalty ${trap_pen} ${BASE_ARGS}
    done
done

for m in easy medium hard; do
    for s in "${SEEDS[@]}"; do
        python train.py --exp-name map_${m} \
            --outdir results/map/${m}/seed_${s} \
            --seed ${s} \
            --map-name ${m} ${BASE_ARGS}
    done
done

echo "Core experiments finished. Starting report experiments."

# ==========================================
# 5. Q-Learning vs SARSA on hard map
# ==========================================
echo "5/6 Q-Learning vs SARSA"
for algo in q_learning sarsa; do
    for s in "${REPORT_SEEDS[@]}"; do
        python train.py --exp-name algorithm_${algo} \
            --algorithm ${algo} \
            --outdir results/algorithm_comparison/${algo}/seed_${s} \
            --seed ${s} \
            --map-name hard \
            --step-penalty -0.1 --trap-penalty -5.0 ${REPORT_ARGS}
    done
done

# ==========================================
# 6. Trap Penalty / Risk Sensitivity
# ==========================================
echo "6/6 Trap Penalty Risk Sensitivity"
for algo in q_learning sarsa; do
    for trap_abs in 5 10 30; do
        trap_penalty="-${trap_abs}"
        for s in "${REPORT_SEEDS[@]}"; do
            python train.py --exp-name trap_penalty_${algo}_${trap_abs} \
                --algorithm ${algo} \
                --outdir results/trap_penalty/${algo}/trap_${trap_abs}/seed_${s} \
                --seed ${s} \
                --map-name risk \
                --step-penalty -0.1 --trap-penalty ${trap_penalty} ${REPORT_ARGS}
        done
    done
done

echo "Report experiments finished. Use python visualize_experiments.py to aggregate comparison figures."

# ==========================================
# 7. SARSA Hyperparameter Tuning
# ==========================================
echo "7/7 SARSA Hyperparameter Tuning"

for a in 0.05 0.1 0.5; do
    for s in "${REPORT_SEEDS[@]}"; do
        python train.py --exp-name sarsa_alpha_${a} \
            --algorithm sarsa \
            --outdir results/sarsa_alpha_comparison/alpha_${a}/seed_${s} \
            --seed ${s} \
            --alpha ${a} ${BASE_ARGS}
    done
done

for g in 0.5 0.8 0.95 0.99; do
    for s in "${SEEDS[@]}"; do
        python train.py --exp-name sarsa_gamma_${g} \
            --algorithm sarsa \
            --outdir results/sarsa_gamma_comparison/gamma_${g}/seed_${s} \
            --seed ${s} \
            --gamma ${g} ${BASE_ARGS}
    done
done

for d in 0.95 0.99 0.995; do
    for s in "${REPORT_SEEDS[@]}"; do
        python train.py --exp-name sarsa_epsilon_decay_${d} \
            --algorithm sarsa \
            --outdir results/sarsa_epsilon_comparison/decay_${d}/seed_${s} \
            --seed ${s} \
            --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay ${d} ${BASE_ARGS}
    done
done

echo "SARSA tuning experiments finished."
