using UnityEngine;
using System.Collections.Generic;
using System.Security.Cryptography;

public class ExecutionManager : MonoBehaviour
{
    public static ExecutionManager Instance { get; private set; }

    private readonly List<Predator> _coalitionBuffer = new List<Predator>(16);
    private readonly List<Predator> _acceptedBuffer = new List<Predator>(16);
    private readonly List<Predator> _workingPartnersBuffer = new List<Predator>(16);
    
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }
    public void Initialize() { Debug.Log("ExecutionManager Initialized"); }

#region Predator Functions

    public void ExecuteDecision(Predator predator)
    {
        string action = predator.decision.actionType;
        Prey huntTarget = null;
        Debug.Log("----- Entered Execution -----");
        switch(action)
        {   
            case "Rest":
                executeRest(predator);
                break;
            case "Move":
                executeMove(predator);
                break;
            case "Solo Hunt":
                Debug.Log("----- Entered Solo Hunt EXE -----");
                huntTarget = predator.decision.targetPrey;
                executeHunt(predator, huntTarget); 
                break;
            case "Coalition Hunt":
                Debug.Log("----- Entered Coalition EXE -----");
                huntTarget = predator.decision.targetPrey;
                executeCoalitionNegotiation(predator, huntTarget);
                executeHunt(predator, huntTarget);  
                break;
            case "_":
                break;
        }
    }

    private void executeMove(Predator predator)
    {
        // UPDATE DATA
        predator.q = predator.decision.targetQ;
        predator.r = predator.decision.targetR;
        predator.food -= SimulationConfig.C_MOVE;

        // PRESENTATION
        Vector3 targetPos = HexGridManager.Instance.GetRandomPositionAtNode(predator.q, predator.r);
        if (predator.TryGetComponent(out LiveAgentMotor motor))
        {
            motor.SetNewDestination(targetPos);
        }
    }

    private void executeHunt(Predator predator, Prey targetPrey)
    {
        Decision decision = predator.decision;

        // Determine hunt success cleanly via Unity's random float [0.0, 1.0]
        bool success = Random.value < decision.huntSuccess;

        // 1. Reset the shared class buffer from the previous linear operation
        _workingPartnersBuffer.Clear();

        // 2. Populate the shared memory space synchronously
        _workingPartnersBuffer.Add(predator);
        if (decision.partners != null) { _workingPartnersBuffer.AddRange(decision.partners); }

        // 3. High-performance index-bound loop (Zero GC allocations, zero iterator overhead)
        for (int i = 0; i < _workingPartnersBuffer.Count; i++)
        {
            Predator p = _workingPartnersBuffer[i];
            Decision pDecision = p.decision;
            float utility = 0f;

            if (success)
            {
                float reward = pDecision.finalShare * targetPrey.reward;
                utility = reward - p.huntCost;
            }
            else
            {
                utility = -p.huntCost;
            }

            p.food += utility;
        }
    }

    private void executeCoalitionNegotiation(Predator initiator, Prey targetPrey)
    {
        
        List<Predator> originalPartners = initiator.decision.partners;

        Debug.Log($"originalPartners: {originalPartners.Count}");

        _workingPartnersBuffer.Clear();
        if (originalPartners != null) { _workingPartnersBuffer.AddRange(originalPartners); }

        while (true)
        {
            Debug.Log("#### Bargain Round ####");
            // 1. Build the active coalition in our reusable buffer
            _coalitionBuffer.Clear();
            _coalitionBuffer.Add(initiator);
            _coalitionBuffer.AddRange(_workingPartnersBuffer);

            int coalitionSize = _coalitionBuffer.Count;

            Debug.Log($"_coalitionBuffer: {_coalitionBuffer.Count}");

            // 2. Perform math lookups using pre-allocated lists
            float totalEC = Compute.ECValue(_coalitionBuffer);
            float coalitionSuccess = Compute.PCaptureValue(targetPrey.pSolo, totalEC, targetPrey.beta, coalitionSize);

            Debug.Log($"_coalitionBuffer success: {coalitionSuccess}");

            // This dictionary allocation remains dynamic unless optimized inside GetSplitShares,
            // but passing our optimized buffer minimizes internal list creation.
            Dictionary<Predator, float> coalitionShares = SimulationManager.Instance.GetSplitShares(initiator, _coalitionBuffer);

            // 3. Evaluate acceptance into a separate reusable buffer
            _acceptedBuffer.Clear();
            for (int i = 0; i < _workingPartnersBuffer.Count; i++)
            {
                Predator p = _workingPartnersBuffer[i];
                if (SimulationManager.Instance.GetCoalitionChecks(p, coalitionSuccess, coalitionShares[p], targetPrey))
                {
                    _acceptedBuffer.Add(p);
                }
            }

            // --- STABILITY CONDITION ---
            // If the number of accepted predators equals our current working group, stability achieved!
            if (_acceptedBuffer.Count == _workingPartnersBuffer.Count) { break; }

            // --- SOLO ESCAPE CONDITION ---
            // If everyone rejected, immediately collapse back to a Solo Hunt
            if (_acceptedBuffer.Count == 0)
            {
                DecisionManager.Instance.UpdateDecision(initiator,
                    actionType: "Solo Hunt",
                    huntSuccess: Compute.PCaptureValue(targetPrey.pSolo, initiator.merit, targetPrey.beta, 1),
                    finalShare: 1.0f,
                    partners: null
                );
                return;
            }

            // --- SHRINK STEP CONDITION ---
            // Copy accepted group over to become the next iteration's working group
            _workingPartnersBuffer.Clear();
            _workingPartnersBuffer.AddRange(_acceptedBuffer);            
        }

        // --- FINALIZATION ---
        // Rebuild the final stable group one last time to get finalized success ratios & splits
        _coalitionBuffer.Clear();
        _coalitionBuffer.Add(initiator);
        _coalitionBuffer.AddRange(_workingPartnersBuffer);

        float finalTotalEC = Compute.ECValue(_coalitionBuffer);
        float finalCoalitionSuccess = Compute.PCaptureValue(targetPrey.pSolo, finalTotalEC, targetPrey.beta, _coalitionBuffer.Count);
        Dictionary<Predator, float> finalCoalitionShares = SimulationManager.Instance.GetSplitShares(initiator, _coalitionBuffer);

        // Explicitly package the stable partner choices into a persistent list for the initiator's long-term memory
        List<Predator> finalStablePartners = new List<Predator>(_workingPartnersBuffer);

        // --- Update Initiator ---
        DecisionManager.Instance.UpdateDecision(initiator,
            huntSuccess: finalCoalitionSuccess,
            finalShare: finalCoalitionShares[initiator],
            partners: finalStablePartners
        );

        // --- Update Partners ---
        for (int i = 0; i < _workingPartnersBuffer.Count; i++)
        {
            Predator p = _workingPartnersBuffer[i];
            DecisionManager.Instance.UpdateDecision(p,
                actionType: "Coalition Hunt",
                huntSuccess: finalCoalitionSuccess,
                finalShare: finalCoalitionShares[p]
            );
            p.hasActed = true;
        }                                        
    }

    private void executeRest(Predator predator)
    {
        // Placeholder function for the idle Animation, probably setting camp
    }

#endregion

#region Prey Functions

    public void UpdatePrey(List<Prey> preyList, int currentTick)
    {
        foreach (Prey prey in preyList)
        {
            if (prey == null || prey.isAlive==false)
                continue;
            
            // Escape Movement
            if (prey.targeted) { ExecutePreyMovement(prey); }
            // Periodic Roaming
            if(currentTick % SimulationConfig.PREY_TICKS_PER_ROAM == 0) { ExecutePreyMovement(prey); }

            prey.targeted = false;

        }
    }

    private void ExecutePreyMovement(Prey prey)
    {
        // LOGIC
        List<Vector2Int> neighborHexes = HexGridManager.Instance.GetNeighborHexes(prey.q, prey.r);
        if (neighborHexes.Count == 0) return;

        // UPDATE DATA
        Vector2Int newHexCoords = neighborHexes[Random.Range(0, neighborHexes.Count)];

        prey.q = newHexCoords.x;
        prey.r = newHexCoords.y;

        // PRESENTATION
        Vector3 targetPos = HexGridManager.Instance.GetRandomPositionAtNode(prey.q, prey.r);
        if (prey.TryGetComponent(out LiveAgentMotor motor))
        {
            motor.SetNewDestination(targetPos);
        }

    }

#endregion


}
