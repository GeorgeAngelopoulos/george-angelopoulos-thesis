using UnityEngine;

[System.Serializable]
public struct HexCoordinates
{
    public int X { get; private set; }
    public int Z { get; private set; }

    public HexCoordinates (int x, int z)
    {
        X = x;
        Z = z;
    }
}
