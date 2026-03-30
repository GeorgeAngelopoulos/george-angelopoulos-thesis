using UnityEngine;

public abstract class Agent : MonoBehaviour
{

    public int agentID;

    [Header("Stats")]
    public float speed;
    public float health;

    [Header("State")]
    public Vector3 currentPosition;
    public bool isAlive = true;

    protected virtual void Start()
    {
        
    }


    protected virtual void Update()
    {
        
    }
}
