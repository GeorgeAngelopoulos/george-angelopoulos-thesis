using UnityEngine;

public abstract class Agent : MonoBehaviour
{
    [Header("Base Variables")]
    public int id;
    public int q; // grid position
    public int r; // grid position

    public bool isAlive = true;

    // --- Initialization ---
    public virtual void Initialize(int id, int q, int r)
    {
        this.id = id;
        this.q = q;
        this.r = r;
    }
}
