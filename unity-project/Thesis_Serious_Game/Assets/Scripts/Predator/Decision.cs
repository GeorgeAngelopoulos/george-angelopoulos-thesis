using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public struct Decision
{
    public string actionType;
    public float utility;
    public int targetQ;
    public int targetR;
    public List<Predator> partners;
    public float finalShare;
    public Prey targetPrey;
    public float huntSuccess;
}
