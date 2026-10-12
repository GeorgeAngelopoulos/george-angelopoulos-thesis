using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public static class Compute
{
    #region Value Computes

    public static float DemandEffValue(float food, float baseDemand, float huntCost)
    {   
        if (huntCost != 0)
        {
            // Ensure food does not drop to zero
            if (food <= 0.001f) 
                food = 0.001f;

            return baseDemand * (1f + Mathf.Exp(-food / huntCost));
        }
        return baseDemand; // Fallback if cost is 0
    }
    
    public static float PCaptureValue(float pSolo, float collectiveEff, float beta, int predatorCount)
    {
        float b = (predatorCount > 1) ? beta : 1.0f;
        return 1.0f - Mathf.Pow(1.0f - pSolo, collectiveEff * b);
    }
    
    public static float ECValue(List<Predator> predators)
    {
        float sum = 0;
        foreach (var p in predators)
        {
            sum += p.merit;
        }
        return sum;
    }
    
    public static float ExpUtilityValue(float pCapture, float xCoal, float reward, float huntCost)
    {
        return pCapture * (xCoal * reward) - huntCost;
    }

    #endregion

    #region Boolean Checks

    public static bool UtilityCheck(Predator pred, float coalitionSuccess, float share, Prey prey)
    {
        float soloSuccess = PCaptureValue(prey.pSolo, pred.merit, prey.beta, 1);

        float soloUtility = ExpUtilityValue(soloSuccess, 1.0f, prey.reward, pred.huntCost);
        float groupUtility = ExpUtilityValue(coalitionSuccess, share, prey.reward, pred.huntCost);
        
        return groupUtility >= soloUtility;
    }
    
    public static bool BargainingCheck(Predator pred, float share)
    {
        return share >= DemandEffValue(pred.food, pred.baseDemand, pred.huntCost);
    }

    #endregion

    #region Coalition Share Computes

    public static float EgalitarianSplitValue(List<Predator> coalition)
    {
        int coalitionSize = coalition.Count;
        return 1f / coalitionSize;
    }

    public static float MeritocraticSplitValue(Predator predator, List<Predator> coalition)
    {
        float coalitionMi = 0f;
        foreach (var pr in coalition) { coalitionMi += pr.merit; }
        return predator.merit / coalitionMi;
    }

    public static float AltruisticSplitValue(Predator predator, List<Predator> coalition)
    {
        float demand = DemandEffValue(predator.food, predator.baseDemand, predator.huntCost);
        float coalitionDemand = 0f;

        foreach (var pr in coalition) 
        { 
            float prDemand = DemandEffValue(pr.food, pr.baseDemand, pr.huntCost);
            coalitionDemand += prDemand; 
        }
        return demand / coalitionDemand;
    }

    public static float AsymmetricSplitValue(Predator predator, Predator initiator, List<Predator> coalition)
    {
        int coalitionSize = coalition.Count;
        float partnersShare = (1f - SimulationConfig.LEADER_SHARE) / coalitionSize;

        if (predator == initiator)
        {
            return SimulationConfig.LEADER_SHARE + partnersShare;
        }
        else
        {
            return partnersShare;
        }
    }

    #endregion

}
