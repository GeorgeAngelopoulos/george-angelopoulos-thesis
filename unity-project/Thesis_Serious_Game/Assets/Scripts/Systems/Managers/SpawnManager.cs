using UnityEngine;

public class SpawnManager : MonoBehaviour
{
    public static SpawnManager Instance { get; private set; }

    [Header("Core References")]
    [SerializeField] private Transform agentParent;
    [SerializeField] private int nextPreyID = 0;
    [SerializeField] private int nextPredatorID = 0;

    [Header("Prefabs")]
    [SerializeField] private GameObject predatorPrefab;
    [SerializeField] private GameObject harePrefab;
    [SerializeField] private GameObject stagPrefab;
    [SerializeField] private GameObject mammothPrefab;

    [Header("Spawn Settings")]
    [SerializeField] private Vector2Int caveHex; // Starting hex (player base)

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }
    public void Initialize() { Debug.Log("SpawnManager Initialized"); }

    private void Start()
    {
        SpawnPreyGroupInHex(0,0);
    }

#region Predator Spawning (Player Action)

    public Predator SpawnPredator(PredatorBuildData build)
    {
        // 2. Instantiate prefab
        GameObject predatorObject = Instantiate(predatorPrefab, Vector3.zero, Quaternion.identity);

        // 2. Add Predator component
        Predator predator =  predatorObject.AddComponent<Predator>();

        // 3. Spawn from Cave Hex (temporary 0.0)
        int caveQ = 0;
        int caveR = 0;

        // Optional world placement for now
        predatorObject.transform.position = Vector3.zero;

        // 4. Initialize Predator Component
        predator.InitializePredator(nextPredatorID, caveQ, caveR, build);
        nextPredatorID++;

        // Determine color based on your 4 tactics
        Color tacticColor = Color.white;
        switch (predator.tactic)
        {
            case "Egalitarian":  tacticColor = Color.green; break;
            case "Meritocratic": tacticColor = Color.blue;  break;
            case "Altruistic":   tacticColor = Color.red;  break;
            case "Asymmetric":   tacticColor = Color.gold;   break;
        }

        // Search the spawned clothing item for its renderer to swap the material color tint
        Renderer renderer = predatorObject.GetComponentInChildren<Renderer>();
        if (renderer != null)
        {
            // Using .material automatically creates a local instance copy so it doesn't change your project asset files
            renderer.material.color = tacticColor; 
        }

        // 5. Register into simulation
        SimulationManager.Instance.RegisterPredator(predator);

        return predator;
    }

#endregion

#region Prey Spawning (on Hex Explore)

    public void SpawnPreyGroupInHex(int q, int r)
    {
        int groupSize = Random.Range(1, SimulationConfig.MAX_PREY_PER_EXPLORED_HEX+1);

        for (int i = 0; i < groupSize; i++)
        {
            PreyType type = GetRandomPreyType();
            SpawnPrey(q, r, type);
        }
    }
    public void SpawnPrey(int q, int r, PreyType type)
    {
        // 1. Get correct prefab
        GameObject prefab = GetPreyPrefab(type);
        if (prefab == null)
        {
            Debug.LogError("No prefab found for prey type: " + type);
            return;
        }

        // 2. Instantiate prefab
        GameObject preyObject = Instantiate(prefab, HexGridManager.Instance.GetRandomPositionAtNode(q, r), Quaternion.identity);

        // 3. Get Prey component (already on prefab)
        Prey prey = preyObject.GetComponent<Prey>();

        if (prey == null)
        {
            Debug.LogError("Prefab missing Prey component.");
            return;
        }

        // 4. Initialize runtime data only
        prey.InitializePrey(nextPreyID, q, r, type);

        nextPreyID++;

        // 5. Register in simulation
        SimulationManager.Instance.RegisterPrey(prey);  
    }

#endregion

#region Despawing (On Recall / Death)

    public void DespawnPredator(Predator predator)
    {
        SimulationManager.Instance.UnregisterPredator(predator);
        //Placeholder for Death Animation()
        Destroy(predator.gameObject);    
    }
    public void DespawnPrey(Prey prey)
    {
        SimulationManager.Instance.UnregisterPrey(prey);
        //Placeholder for Death Animation()
        Destroy(prey.gameObject);      
    }

#endregion

#region Helper Functions

    private PreyType GetRandomPreyType()
    {
        int hare = SimulationConfig.HARE_WEIGHT;
        int stag = SimulationConfig.STAG_WEIGHT;
        int mammoth = SimulationConfig.MAMMOTH_WEIGHT;

        int total = hare + stag + mammoth;

        if (total <= 0)
            return PreyType.Hare;

        int roll = Random.Range(0, total);

        if (roll < hare)
            return PreyType.Hare;

        if (roll < hare + stag)
            return PreyType.Stag;

        return PreyType.Mammoth;
    } 
    private GameObject GetPreyPrefab(PreyType type)
    {
        switch (type)
        {
            case PreyType.Hare: return harePrefab;
            case PreyType.Stag: return stagPrefab;
            case PreyType.Mammoth: return mammothPrefab;
            default:
                Debug.LogWarning("Unknown prey type: " + type);
                return harePrefab;
        }
    }

#endregion

}
