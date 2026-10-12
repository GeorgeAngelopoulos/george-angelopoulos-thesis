using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class PredatorCreationMenu : MonoBehaviour
{
    [Header("Zero-Sum Budget Settings")]
    [SerializeField] private float availablePointsPool = 0f;

    // Hardcoded simulation baselines for delta tracking
    private const float BASELINE_MERIT = 1.5f;
    private const float BASELINE_DEMAND = 0.10f;
    private const float BASELINE_HUNT = 5.0f;

    [Header("Current Character Characteristics")]
    [SerializeField] private float merit = 1.5f;
    [SerializeField] private float baseDemand = 0.10f;
    [SerializeField] private float huntCost = 5.0f;
   
    private string[] tactics = { "Egalitarian", "Meritocratic", "Altruistic", "Asymmetric" };
    private int currentTacticIndex = 0;

    [Header("UI Text References")]
    [SerializeField] private TMP_Text budgetText;
    [SerializeField] private TMP_Text meritValueText;
    [SerializeField] private TMP_Text baseDemandValueText;
    [SerializeField] private TMP_Text huntCostValueText;
    [SerializeField] private TMP_Text tacticDisplayText;
    [SerializeField] private Button createPredatorButton;

    private void Start()
    {
        UpdateCreationUI();
        UpdateTacticUI(); // Ensure the starting tactic displays instantly on play
    }

    // --- CHARACTERISTIC BUTTON ACTIONS (Pass 1 for Up, -1 for Down) ---

    public void AdjustMerit(int direction)
    {
        merit = Mathf.Clamp(merit + (direction * 0.5f), 0.5f, 3.0f);
        UpdateCreationUI();
    }

    public void AdjustBaseDemand(int direction)
    {
        baseDemand = Mathf.Clamp(baseDemand + (direction * 0.05f), 0.1f, 0.4f);
        UpdateCreationUI();
    }

    public void AdjustHuntCost(int direction)
    {
        huntCost = Mathf.Clamp(huntCost + (direction * 5.0f), 5.0f, 20.0f);
        UpdateCreationUI();
    }

    // --- TACTIC BUTTON ACTIONS ---

    public void CycleTactic(int direction)
    {
        currentTacticIndex += direction;
        
        // Wrap around logic if clicking past array edges
        if (currentTacticIndex >= tactics.Length) currentTacticIndex = 0;
        if (currentTacticIndex < 0) currentTacticIndex = tactics.Length - 1;

        UpdateTacticUI();
    }

    private void UpdateTacticUI()
    {
        tacticDisplayText.text = tactics[currentTacticIndex];
    }

    // --- CORE REFRESH & BALANCE SYSTEM ---

    private void UpdateCreationUI()
    {
        // 1. Calculate how many steps the player has drifted away from the baseline
        float meritStepsFromBase = (merit - BASELINE_MERIT) / 0.5f;
        float demandStepsFromBase = (baseDemand - BASELINE_DEMAND) / 0.05f;
        float huntStepsFromBase = (huntCost - BASELINE_HUNT) / 5.0f;

        // 2. Balance pool conversion math
        float meritCost = meritStepsFromBase * 20f;
        float demandRefund = demandStepsFromBase * 10f;
        float huntRefund = huntStepsFromBase * 15f;

        availablePointsPool = 0f - meritCost + demandRefund + huntRefund;

        // 3. Update Characteristic Display Text strings
        meritValueText.text = $"{merit:F1}";
        baseDemandValueText.text = $"{baseDemand:F2}";
        huntCostValueText.text = $"{huntCost:F0} Food";

        // Display positive values with a plus sign for visual clarity
        string poolSign = availablePointsPool >= 0 ? "+" : "";
        budgetText.text = $"Balance Pool: {poolSign}{availablePointsPool:F0}";

        // 4. Budget Enforcement Guardrail
        if (availablePointsPool < 0)
        {
            budgetText.color = Color.red;
            createPredatorButton.interactable = false;
        }
        else
        {
            budgetText.color = Color.white;
            createPredatorButton.interactable = true;
        }
    }

    public void OnCreatePredatorPressed()
    {
        PredatorBuildData build = new PredatorBuildData("Custom", merit, baseDemand, huntCost, tactics[currentTacticIndex]);
        SpawnManager.Instance.SpawnPredator(build);
        GameManager.Instance.SpendFood(100f);  
    }
}
