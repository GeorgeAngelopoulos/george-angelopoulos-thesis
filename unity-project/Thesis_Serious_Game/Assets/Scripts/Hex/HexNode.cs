using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public struct HexNode
{
    public int q; 
    public int r; 

    // Cube Coordinate S is derived: s = -q - r
    public int s => -q - r;

    public Vector3 worldPosition;
    
    public bool isWalkable;

    public List<Predator> residentPredators;
    public List<Prey> residentPrey;

    public HexNode(int q, int r, Vector3 worldPosition, bool walkable)
    {
        this.q = q;
        this.r = r;
        this.worldPosition = worldPosition;
        isWalkable = walkable;
        residentPredators = new List<Predator>();
        residentPrey = new List<Prey>();
    }
}
