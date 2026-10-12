using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class UIManager : MonoBehaviour
{
    public static UIManager Instance { get; private set; }
    public enum DashboardState { None, Creation, Recall, Select}

    [Header("Current UI State")]
    [SerializeField] private DashboardState currentState = DashboardState.None;

    [Header("UI Panels")]
    [SerializeField] private GameObject StatsUIPanel;


    [Header("Dashboard Panels")]
    [SerializeField] private GameObject DashboardFrame;
    [SerializeField] private GameObject creationContent;
    [SerializeField] private GameObject recallContent;
    [SerializeField] public GameObject selectContent;       

    [Header("Text Displays")]
    public TextMeshProUGUI foodText;
    public TextMeshProUGUI perDayFoodText;
    public TextMeshProUGUI popText;
    public TextMeshProUGUI predatorText;
    public TextMeshProUGUI tickText;

    [Header("Text Messages")]
    public TextMeshProUGUI messageText;

    // --- Internal state tracking (prevents redundant updates) ---
    private float lastFood = -1.0f;
    private int lastPop = -1;
    private int lastPredatorCount = 0;

    private float lastPerDayFood = -1.0f;


    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }
    public void Initialize() { Debug.Log("UIManager Initialized"); }

    private void InitializationUI()
    {
        UpdateUIFood(GameManager.Instance.GetFoodSupply(), 0);
        UpdateUIPop(GameManager.Instance.GetPopulation());
        UpdateUIPredatorCount(GameManager.Instance.GetActivePredatorCount());
        UpdateUITime(TickManager.Instance.CurrentDay, TickManager.Instance.CurrentHour);

        creationContent.SetActive(false);
        recallContent.SetActive(false);
        selectContent.SetActive(false);
        DashboardFrame.SetActive(false);
        currentState = DashboardState.None;
    }

#region Tick Handlers

    private void Start() 
    { 
        TickManager.Instance.OnTick += HandleTick;
        InitializationUI(); 
    }
    private void OnDestroy()
    {
        if (TickManager.Instance != null)
            TickManager.Instance.OnTick -= HandleTick;
    }
    private void HandleTick(int tick)
    {
        UpdateUITime(TickManager.Instance.CurrentDay, TickManager.Instance.CurrentHour);
    }

#endregion

#region Update UI Text Functions
    public void UpdateUIFood(float value, float perDayFood)
    {
        if (value == lastFood && perDayFood == lastPerDayFood) return;

        lastFood = value;
        lastPerDayFood = perDayFood;

        // 2. Determine color tag and optional sign prefix based on the value
        string colorTag;
        string signPrefix = "";

        if (perDayFood > 0f)
        {
            colorTag = "<color=green>";
            signPrefix = "+"; // Adds a "+" sign for visual clarity: (+5.50)
        }
        else if (perDayFood < 0f)
        {
            colorTag = "<color=red>";
            // No minus prefix needed because float formatting (:F2) preserves the native '-' sign
        }
        else
        {
            colorTag = "<color=white>"; // Neutral color if growth is exactly 0
        }

        foodText.text = $"{value:F2}";

        perDayFoodText.text = $"{colorTag}{signPrefix}{perDayFood:F2}</color>";
    }

    public void UpdateUIPop(int value)
    {
        if (value == lastPop) return;

        lastPop = value;
        popText.text = $"{value}";
    }

    public void UpdateUIPredatorCount(int count)
    {
        if (count == lastPredatorCount) return;

        lastPredatorCount = count;
        predatorText.text = $"{count}";
    }

    public void UpdateUITime(int day, int hour)
    {
        tickText.text = "Day " + day + " - " + hour.ToString("00") + ":00";
    }
    
#endregion

#region Time Button Functions

    public void OnPause() 
    { 
        TickManager.Instance.TogglePause(true);
    }

    public void OnPlay()
    { 
        TickManager.Instance.TogglePause(false);
        TickManager.Instance.SetSpeed(1f);
    }

    public void OnSpeed2x()
    { 
        TickManager.Instance.TogglePause(false);
        TickManager.Instance.SetSpeed(2f); 
    }

    public void OnSpeed3x()
    {
        TickManager.Instance.TogglePause(false);
        TickManager.Instance.SetSpeed(3f);    
    }

#endregion

#region UI Button Functions

    public void onChangeDashboardState(int stateIndex)
    {
        DashboardState targetState = (DashboardState)stateIndex;
        // 1. If clicking the already active tab -> Close the dashboard entirely
        if(currentState == targetState && currentState != DashboardState.Select )
        {
            CloseDashboard();
            return;
        }
        // 2. If the dashboard was hidden, slide or turn it on
        if (currentState == DashboardState.None) { OpenDashboard(); }
        // 3. Swap the inner content fragments instantly
        SwitchDashboardContent(targetState);
        // Update our tracker
        currentState = targetState;
    }

    public void onOpenStatisticsWindow()
    {
        // Open the Statistics Window Panel
        OnPause();
    }

    public void onCloseStatisticsWindow()
    {
        // Close the Statistics Window Panel
        OnPlay();
    }

#endregion

#region Dashboard Functions

    private void SwitchDashboardContent( DashboardState activeState)
    {
        // Turn everything off first to ensure no overlap anomalies
        creationContent.SetActive(false);
        recallContent.SetActive(false);
        selectContent.SetActive(false);

        switch (activeState)
        {
            case DashboardState.Creation:
                creationContent.SetActive(true);
                break;

            case DashboardState.Recall:
                recallContent.SetActive(true);
                break;

            case DashboardState.Select:
                selectContent.SetActive(true);
                break;
        }
    }

    private void OpenDashboard() { DashboardFrame.SetActive(true); }

    private void CloseDashboard()
    {
        DashboardFrame.SetActive(false);
        currentState = DashboardState.None;
    }

#endregion

    public void ShowMessage(string msg)
    {
        messageText.text = msg;
        CancelInvoke();
        Invoke(nameof(ClearMessage), 2f);
    }

    private void ClearMessage()
    {
        messageText.text = "";
    }

}
