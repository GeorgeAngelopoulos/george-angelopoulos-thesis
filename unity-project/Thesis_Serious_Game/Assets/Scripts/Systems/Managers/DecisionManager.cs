using UnityEngine;
using System.Collections.Generic;

public class DecisionManager : MonoBehaviour
{
    public static DecisionManager Instance { get; private set; }

    private readonly float gamma = 0.9f;
    
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }
    public void Initialize() { Debug.Log("DecisionManager Initialized"); }

    public void FormBestDecision(Predator predator)
    {
        var (predatorsPresent, preyPresent) = SimulationManager.Instance.GetNearbyAgents(predator);

        float bestUtility = float.NegativeInfinity;
        Decision bestDecision = default(Decision);

        bestDecision.targetQ = predator.q;
        bestDecision.targetR = predator.r;
        
        float utility = ExpectedUtilityRest();
        if (utility > bestUtility)
        {
            bestUtility = utility;

            bestDecision.actionType = "Rest";
            bestDecision.utility = utility;
        }
        var (moveUtility, q, r) = ExpectedUtilityMove(predator);
        if (moveUtility > bestUtility)
        {
            bestUtility = moveUtility;

            bestDecision.actionType = "Move";
            bestDecision.utility = moveUtility;
            bestDecision.targetQ = q;
            bestDecision.targetR = r;  
        }

        if(preyPresent != null && preyPresent.Count > 0)
        {
            var (soloUtility, targetPrey) = ExpectedUtilitySoloHunt(predator, preyPresent);
            if (soloUtility > bestUtility)
            {
                bestUtility = soloUtility;

                bestDecision.actionType = "Solo Hunt";
                bestDecision.utility = soloUtility;
                bestDecision.targetPrey = targetPrey;
                bestDecision.finalShare = 1.0f;
                bestDecision.huntSuccess = SimulationManager.Instance.GetHuntSuccess(predator, null, targetPrey);
            }         
        }
        

        if(preyPresent != null && preyPresent.Count > 0 && predatorsPresent != null && predatorsPresent.Count > 0)
        {
            var (coalUtility, targetPrey, partners, share) = ExpectedUtilityCoalitionHunt(predator, predatorsPresent, preyPresent);
            if (coalUtility > bestUtility)
            {
                bestUtility = coalUtility;

                bestDecision.actionType = "Coalition Hunt";
                bestDecision.utility = coalUtility;
                bestDecision.targetPrey = targetPrey;
                bestDecision.partners = partners;
                bestDecision.finalShare = share;
                bestDecision.huntSuccess = SimulationManager.Instance.GetHuntSuccess(predator, partners, targetPrey);
            }
        }
        
        if (bestDecision.targetPrey != null)
        {
            bestDecision.targetPrey.targeted = true;
        }
        
        predator.decision = bestDecision;
        predator.hasActed = true;
        Debug.Log($"Predator's {predator} action = {predator.decision.actionType}");    
    }

    public void UpdateDecision(Predator predator, string actionType = null, float? utility = null, int? targetQ = null, int? targetR = null, List<Predator> partners = null, float? finalShare = null, Prey targetPrey = null, float? huntSuccess = null)
    {
        Decision currentDecision = predator.decision;

        if (actionType != null)   currentDecision.actionType = actionType;
        if (utility != null)      currentDecision.utility = utility.Value;
        if (targetQ != null)      currentDecision.targetQ = targetQ.Value;
        if (targetR != null)      currentDecision.targetR = targetR.Value;
        if (partners != null)     currentDecision.partners = partners;
        if (finalShare != null)   currentDecision.finalShare = finalShare.Value;
        if (targetPrey != null)   currentDecision.targetPrey = targetPrey;
        if (huntSuccess != null)  currentDecision.huntSuccess = huntSuccess.Value;

        predator.decision = currentDecision;
    }

    #region Expected Utility Calculations

        private float ExpectedUtilityRest() { return 0f; }

        private (float, int, int) ExpectedUtilityMove(Predator predator)
        {   
            List<Vector2Int> neighborHexes = HexGridManager.Instance.GetNeighborHexes(predator.q, predator.r);

            var typeUtilities = new Dictionary<PreyType, float>();

            foreach (var kvp in PreyDatabase.PREY_PROFILES)
            {
                PreyType pType = kvp.Key;     // The PreyType (Hare, Stag, Mammoth)
                var profile = kvp.Value;      // The Tuple values (pSolo, reward, baseBeta)

                float PCapture = Compute.PCaptureValue(profile.pSolo, predator.merit, profile.baseBeta, 1);
                float USolo = Compute.ExpUtilityValue(PCapture, 1.0f, profile.reward, predator.huntCost);

                typeUtilities[pType] = USolo;
            }
            
            float bestUtility = float.NegativeInfinity;
            Vector2Int bestHex = new Vector2Int(predator.q, predator.r);

            foreach (Vector2Int neighbor in neighborHexes)
            {
                float vBestHex = float.NegativeInfinity;

                foreach (var prey in SimulationManager.Instance.preyList)
                {
                    float uSolo = typeUtilities[prey.preyType];

                    // Using the neighbor and prey positions directly via Vector2Int
                    Vector2Int neighborHex = neighbor;
                    Vector2Int preyHex = new Vector2Int(prey.q, prey.r);
                    int distance = HexGridManager.Instance.GetAxialDistance(neighborHex, preyHex);

                    float vAdjHex;
                    if (distance == 0)
                    {
                        vAdjHex = uSolo;
                    }
                    else
                    {
                        float gammaD = Mathf.Pow(gamma, distance);
                        // discounted future reward
                        float discountedReward = gammaD * uSolo;
                        // cumulative movement cost (geometric series)
                        float discountedCost = SimulationConfig.C_MOVE * (1f - gammaD) / (1f - gamma);
                        vAdjHex = discountedReward - discountedCost;
                    }

                    if (vAdjHex > vBestHex)
                    {
                        vBestHex = vAdjHex;
                    }
                }

                // final move utility
                float moveUtility = -SimulationConfig.C_MOVE + gamma * vBestHex;

                if (moveUtility > bestUtility)
                {
                    bestUtility = moveUtility;
                    bestHex = neighbor;
                }
            }

            return (bestUtility, bestHex.x, bestHex.y);
        }

        private (float, int, int) ExpectedUtilityMoveRandom(Predator predator)
        {
            // LOGIC
            List<Vector2Int> neighborHexes = HexGridManager.Instance.GetNeighborHexes(predator.q, predator.r);

            // UPDATE DATA
            Vector2Int newHexCoords = neighborHexes[Random.Range(0, neighborHexes.Count)];

            return (1f, newHexCoords.x, newHexCoords.y);
        }

        private (float, Prey) ExpectedUtilitySoloHunt(Predator predator, List<Prey> preyPresent)
        {
            float bestUtility = float.NegativeInfinity;
            Prey targetPrey = null;

            foreach (Prey prey in preyPresent)
            {
                float pCapture = Compute.PCaptureValue(prey.pSolo, predator.merit, prey.beta, 1);
                float utility = Compute.ExpUtilityValue(pCapture, 1f, prey.reward, predator.huntCost);
                if (utility > bestUtility)
                {
                    bestUtility = utility;
                    targetPrey = prey;
                }
            }
            Debug.Log($"Solo Hunt Exp Utility: {bestUtility}");
            return (bestUtility, targetPrey);
        }

        private (float, Prey, List<Predator>, float) ExpectedUtilityCoalitionHunt(Predator predator, List<Predator> predatorsPresent, List<Prey> preyPresent)
        {
            Debug.Log("----- Entered Exp. Coalition -----");
            float bestUtility = float.NegativeInfinity;
            Prey targetPrey = null;

            List<Predator> coalition = new List<Predator> { predator };
            coalition.AddRange(predatorsPresent);

            int coalitionSize = coalition.Count;
            
            float totalEC = Compute.ECValue(coalition);

            float initiatorShare = 0f;
            initiatorShare = SimulationManager.Instance.GetSplitShares(predator, coalition)[predator];
            Debug.Log($"Potential IN share: {initiatorShare}");
            foreach (Prey prey in preyPresent)
            {
                float pCapture = Compute.PCaptureValue(prey.pSolo, totalEC, prey.beta, coalitionSize);
                float utility = Compute.ExpUtilityValue(pCapture, initiatorShare, prey.reward, predator.huntCost);

                if (utility > bestUtility)
                {
                    bestUtility = utility;
                    targetPrey = prey;
                }
            }
            Debug.Log($"Coalition Exp Utility: {bestUtility}");
            return (bestUtility, targetPrey, predatorsPresent, initiatorShare);
        }

    #endregion
    
}
