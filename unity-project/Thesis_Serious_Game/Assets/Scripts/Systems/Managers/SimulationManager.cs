using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class SimulationManager : MonoBehaviour
{
    public static SimulationManager Instance { get; private set; }
    
    public List<Predator> predators = new List<Predator>();
    public List<Prey> preyList = new List<Prey>();
    private readonly List<Predator> _workingPredatorBuffer = new List<Predator>(128);
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }
    public void Initialize() { Debug.Log("SimulationManager Initialized"); }

#region Tick Handlers

    private void Start() { TickManager.Instance.OnTick += HandleTick; }
    private void OnDestroy()
    {
        if (TickManager.Instance != null)
            TickManager.Instance.OnTick -= HandleTick;
    }
    private void HandleTick(int tick) { RunSimulationStep(tick); }

#endregion

#region Simulation Step Functions

    private void RunSimulationStep(int tick)
    {

        List<Predator> availPredators = DefineAvailablePredators(predators);

        for (int i = 0; i < availPredators.Count; i++)
        {
            Predator predator = availPredators[i];
            
            // Skip the agent if it was already updated as a partner 
            // by a prior agent's coalition execution earlier this tick
            if (!predator.isAlive || predator.hasActed) continue;

            DecisionManager.Instance.FormBestDecision(predator);
            ExecutionManager.Instance.ExecuteDecision(predator);
            UpdateFoodReturn(predator);
            UpdateAliveState(predator);
        }

        ExecutionManager.Instance.UpdatePrey(preyList, tick);
        ResetCycle();
    }
    private List<Predator> DefineAvailablePredators(List<Predator> predators)
    {   
        // Clear out the previous tick's remaining data while keeping memory allocation intact
        _workingPredatorBuffer.Clear();

        // Use a standard loop to avoid hidden iterator allocations
        for (int i = 0; i < predators.Count; i++)
        {
            Predator predator = predators[i];
            if (predator.isAlive && !predator.hasActed) 
            { 
                _workingPredatorBuffer.Add(predator); 
            }
        }

        ShuffleList(_workingPredatorBuffer);

        // Returning the class reference directly with zero dynamic allocation
        return _workingPredatorBuffer;
    }
    private void UpdateFoodReturn(Predator predator)
    {
        float foodReturn = predator.food - SimulationConfig.STARTING_FOOD;
        if (foodReturn > 0)
        {
            predator.food -= foodReturn;
            GameManager.Instance.todayIncome += foodReturn;
        }
    }
    private void UpdateAliveState(Predator predator)
    {
        if (predator.food <= 0)
        { 
            predator.isAlive = false;
            SpawnManager.Instance.DespawnPredator(predator);
        }
    }
    private void ResetCycle()
    {
        foreach(Predator predator in predators)
        {
            predator.hasActed = false;
            predator.decision = default(Decision);
        }

        foreach(Prey prey in preyList)
        {
            prey.targeted = false;
        }
    }

#endregion

#region Agent Registers

    public void RegisterPredator(Predator predator) { predators.Add(predator); }
    public void UnregisterPredator(Predator predator) { predators.Remove(predator); }
    public void RegisterPrey(Prey prey) { preyList.Add(prey); }
    public void UnregisterPrey(Prey prey) { preyList.Remove(prey); }

#endregion

#region Helper Functions

    private void ShuffleList<T>(List<T> list)
    {
        for (int i = 0; i < list.Count; i++)
        {
            int randomIndex = Random.Range(i, list.Count);

            T temp = list[i];
            list[i] = list[randomIndex];
            list[randomIndex] = temp;
        }
    }

    public float GetHuntSuccess(Predator initiator, List<Predator> partners, Prey targetPrey)
    {
        var preds = partners ?? Enumerable.Empty<Predator>();
        float totalEC = initiator.merit + preds.Sum(p => p.merit);
        int coalitionSize = 1 + preds.Count();

        return Compute.PCaptureValue(targetPrey.pSolo, totalEC, targetPrey.beta, coalitionSize);
    }

    public (List<Predator>, List<Prey>) GetNearbyAgents(Predator currentPredator)
    {
        var predatorsPresent = predators.Where(pred => 
        pred.q == currentPredator.q && 
        pred.r == currentPredator.r && 
        pred.isAlive && 
        !pred.hasActed && 
        pred != currentPredator
        ).ToList();

        var preyPresent = preyList.Where(prey => 
        prey.q == currentPredator.q && 
        prey.r == currentPredator.r && 
        prey.isAlive && 
        !prey.targeted
        ).ToList();

        return (predatorsPresent, preyPresent);
    }

    public Dictionary<Predator, float> GetSplitShares(Predator initiator, List<Predator> coalition)
    {
        Debug.Log("------ Entered GetSplitShares ------");
        Debug.Log($"initiator tactic: {initiator.tactic}");
        Debug.Log($"coalition: {coalition.Count}");
        var shares = new Dictionary<Predator, float>();

        foreach (Predator pr in coalition)
        {
            float share = 0f;
            switch (initiator.tactic)
            {
                case "Egalitarian":
                    share = Compute.EgalitarianSplitValue(coalition);
                    break;
                    
                case "Meritocratic":
                    share = Compute.MeritocraticSplitValue(pr, coalition);
                    break;

                case "Altruistic":
                    share = Compute.AltruisticSplitValue(pr, coalition);
                    break;

                case "Asymmetric":
                    share = Compute.AsymmetricSplitValue(pr, initiator, coalition);
                    break;
            }

            shares[pr] = share;
        }

        foreach (var kvp in shares)
        {
            Debug.Log($"Predator: {kvp.Key.name}, Share: {kvp.Value:F2}");
        }

        return shares;
    }

    public bool GetCoalitionChecks(Predator predator, float coalitionSuccess, float predatorShare, Prey targetPrey)
    {
        if (!Compute.BargainingCheck(predator, predatorShare)) {return false;}
        if (!Compute.UtilityCheck(predator, coalitionSuccess, predatorShare, targetPrey)) {return false;}
        return true;
    }

#endregion

}
