using UnityEngine;

public class GameBootstrapper : MonoBehaviour
{
    private void Start() { InitializeSystems(); }
    private void InitializeSystems()
    {
        Debug.Log("=== INITIALIZING MANAGERS ===");

        SimulationManager.Instance.Initialize();
        TickManager.Instance.Initialize();
        SpawnManager.Instance.Initialize();
        DecisionManager.Instance.Initialize();
        ExecutionManager.Instance.Initialize();
        GameManager.Instance.Initialize();
        UIManager.Instance.Initialize();

        Debug.Log("=== ALL MANAGERS INITIALIZED ===");
    }
}
