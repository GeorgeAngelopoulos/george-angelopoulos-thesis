using UnityEngine;
using System.Collections.Generic;

public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    [Header("Resources")]
    [SerializeField] private float foodSupply = 1000.0f;
    [SerializeField] private int population = 20;
    [SerializeField] private int activePredators = 0;

    //[Header("Costs")]
    //[SerializeField] private float exploreCost = 200f;

    [Header("Population System")]
    [SerializeField] private int softCap = SimulationConfig.POP_SOFT_CAP;
    [SerializeField] private float growthRate = SimulationConfig.POP_GROWTH_RATE;

    [Header("Read Only")]
    [SerializeField] private float upkeep = 0f;
    [SerializeField] private float surplus = 0f;
    [SerializeField] private float growth = 0f;

    private HashSet<Vector2Int> exploredHexes = new HashSet<Vector2Int>();
    public float todayIncome = 0f;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }
    public void Initialize() { Debug.Log("GameManager Initialized"); }

#region Tick Handlers

    private void Start() { TickManager.Instance.OnTick += HandleTick; }
    private void OnDestroy()
    {
        if (TickManager.Instance != null)
            TickManager.Instance.OnTick -= HandleTick;
    }
    private void HandleTick(int tick) { RunPopulationProcess(tick); }

#endregion

#region Population Process Functions
    // TODO: MAKE SURE IT WORKS NORMALLY
    public void RunPopulationProcess(int tick)
    {
        CalculateUpkeep();
        CalculateSurplus();
        CalculateGrowth();
        RefreshUI();

        if(tick % TickManager.Instance.ticksPerDay == 0)
        {
            ApplyPopulationChange();
            FinalizeFoodDay();

            CalculateUpkeep();
            CalculateSurplus();
            CalculateGrowth();
            RefreshUI();

        }
    }
    public void CalculateUpkeep() { upkeep = Mathf.Pow(population, SimulationConfig.POP_UPKEEP_RATE); }
    public void CalculateSurplus() { surplus = foodSupply + todayIncome - upkeep; }
    public void CalculateGrowth()
    {
        if (surplus >= 0f)
        {
            growth = growthRate * Mathf.Log(1f + surplus) * (1f - ((float)population / softCap));
        }
        else
        {
            growth = -0.5f * Mathf.Log(1f + Mathf.Abs(surplus));
        }
    }
    public void ApplyPopulationChange()
    {
        // Player-favored thresholds
        if (growth >= 0.75f)
            population++;

        if (growth <= -1.25f)
            population--;

        population = Mathf.Max(1, population);
    }
    private void FinalizeFoodDay()
    {
        foodSupply += todayIncome;
        foodSupply -= upkeep;

        if (foodSupply < 0f)
            foodSupply = 0f;

        todayIncome = 0f;
    }
#endregion

#region Resources

    public float GetFoodSupply() => foodSupply;
    public int GetPopulation() => population;
    public int GetActivePredatorCount() => activePredators;
    public bool HasEnoughFood(float amount) { return foodSupply >= amount;}
    public void AddFood(float amount)
    {
        foodSupply += amount;
        RefreshUI();
    }
    public void SpendFood(float amount)
    {
        foodSupply -= amount;
        RefreshUI();
    }
    public void AddPopulation(int amount)
    {
        population += amount;
        RefreshUI();
    }
    public void RemovePopulation(int amount)
    {
        population -= amount;
        RefreshUI();
    }
    public void AddActivePredator()
    {
        activePredators++;
        RefreshUI();
    }
    public void RemoveActivePredator()
    {
        activePredators = Mathf.Max(0, activePredators - 1);
        RefreshUI();
    }

#endregion

#region Exploration

    //public bool IsHexExplored(Vector2Int hex);

    //public bool CanExploreHex(Vector2Int hex);

    //public bool TryExploreHex(Vector2Int hex);

#endregion

#region Predator Deployment

    public bool TryDeployPredator(PredatorBuildData build)
    {   
        
        float cost = CalculatePredatorCost(build);

        if (!HasEnoughFood(cost))
            return false;

        Predator predator = SpawnManager.Instance.SpawnPredator(build);
        SpendFood(cost);

        if (predator == null)
        {
            AddFood(cost); // refund
            //ShowMessage("Deployment failed.");
            return false;
        }
        AddActivePredator();
        //ShowMessage("Predator deployed.");

        return true;
    }
    public float CalculatePredatorCost(PredatorBuildData build)
    {
        float cost = SimulationConfig.STARTING_FOOD;

        cost += build.merit * SimulationConfig.MERIT_COST_PER_POINT;
        cost += build.baseDemand * SimulationConfig.DEMAND_COST_PER_POINT;
        cost += build.huntCost * SimulationConfig.EFFORT_COST_PER_POINT;

        return cost;  
    }

#endregion

#region Predator Recall

    public bool TryRecallPredator(Predator predator)
    {
        //TODO
        return false;
    }

#endregion

#region UI

    public void RefreshUI()
    {
        if (UIManager.Instance == null)
            return;

        UIManager.Instance.UpdateUIFood(foodSupply, todayIncome - upkeep);
        UIManager.Instance.UpdateUIPop(population);
        UIManager.Instance.UpdateUIPredatorCount(activePredators);
    }

    //public void ShowMessage(string msg);

#endregion

}
