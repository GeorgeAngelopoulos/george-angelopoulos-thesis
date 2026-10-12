using UnityEngine;

[System.Serializable]
public class PredatorBuildData
{
    public string agentType;
    public float merit;
    public float baseDemand;
    public float huntCost;
    public string tactic;

    public PredatorBuildData(string agentType, float merit, float baseDemand, float huntCost, string tactic)
    {
        this.agentType = agentType;
        this.merit = merit;
        this.baseDemand = baseDemand;
        this.huntCost = huntCost;
        this.tactic = tactic;
    }
}