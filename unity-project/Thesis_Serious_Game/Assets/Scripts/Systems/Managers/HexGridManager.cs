using UnityEngine;
using System.Collections.Generic;
using UnityEngine.InputSystem;
using System.Linq;

public class HexGridManager : MonoBehaviour
{
    public static HexGridManager Instance { get; private set; }
    
    [Header("Grid Settings")]
    public int mapRadius = 5;

    [Header("References")]
    public HexMeshGenerator meshGenerator;
    
    public List<HexNode> nodes = new List<HexNode>();
    private static readonly Vector2Int[] HEX_DIRECTIONS =
    {
        new Vector2Int(1, 0), new Vector2Int(1, -1), new Vector2Int(0, -1),
        new Vector2Int(-1, 0), new Vector2Int(-1, 1), new Vector2Int(0, 1)
    };

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }
    public void Initialize() { Debug.Log("HexGridManager Initialized"); }

    void Start()
    {
        GenerateCenteredGrid();
    }
    
    #region Grid Generation

        void GenerateCenteredGrid()
        {
            nodes.Clear();

            // Iterate through a square bounding box that covers the radius
            for (int q = -mapRadius; q <= mapRadius; q++)
            {
                for (int r = -mapRadius; r <= mapRadius; r++)
                {
                    // To keep the grid in a hexagonal shape, we check the hidden 'S' 
                    // coordinate constraint: q + r + s = 0
                    int s = -q - r;

                    if (Mathf.Abs(s) <= mapRadius)
                    {
                        bool walkable;
                        if (q==0 && r == 0) { walkable = true; }
                        else { walkable = Random.value < 0.75f; }
                        nodes.Add(CreateNode(q, r, walkable));
                    }
                }
            }

            meshGenerator.GenerateMesh(nodes.ToArray());
        }
        HexNode CreateNode(int q, int r, bool walkable)
        {
            // Math for Pointy-Topped Axial to World Position
            // Horizontal spacing: q * (size * sqrt(3))
            // Vertical spacing: r * (size * 3/2)
            float x = HexMetrics.innerRadius * 2f * (q + r / 2f);
            float z = HexMetrics.outerRadius * 1.5f * r;

            Vector3 worldPos = new Vector3(x, 0f, z);
            
            // Every node now identifies strictly by its Q and R
            return new HexNode(q, r, worldPos, walkable);
        }

    #endregion

    #region Helper Functions
    
        public bool IsWalkable(int q, int r)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i].q == q && nodes[i].r == r) return nodes[i].isWalkable;
            }
            return false;
        }
        public List<Vector2Int> GetNeighborHexes(int q, int r)
        {
            List<Vector2Int> neighbors = new List<Vector2Int>();
            
            foreach (Vector2Int dir in HEX_DIRECTIONS)
            {
                int targetQ = q + dir.x;
                int targetR = r + dir.y;
                
                if (IsWalkable(targetQ, targetR))
                {
                    neighbors.Add(new Vector2Int(targetQ, targetR));
                }
            }
            return neighbors;
        }
        public int GetAxialDistance(Vector2Int a, Vector2Int b)
        {
            return (Mathf.Abs(a.x - b.x)
                    + Mathf.Abs(a.x + a.y - b.x - b.y)
                    + Mathf.Abs(a.y - b.y)) / 2;
        }
        public HexNode GetNodeAtPoint(Vector3 worldPosition)
        {
            // 1. Convert 3D world position to fractional axial coordinates
            float q = (Mathf.Sqrt(3f) / 3f * worldPosition.x - 1f / 3f * worldPosition.z) / HexMetrics.outerRadius;
            float r = (2f / 3f * worldPosition.z) / HexMetrics.outerRadius;

            // 2. Convert to Cube for rounding
            float x = q;
            float z = r;
            float y = -x - z;

            // 3. Round
            int rx = Mathf.RoundToInt(x);
            int ry = Mathf.RoundToInt(y);
            int rz = Mathf.RoundToInt(z);

            float xDiff = Mathf.Abs(rx - x);
            float yDiff = Mathf.Abs(ry - y);
            float zDiff = Mathf.Abs(rz - z);

            if (xDiff > yDiff && xDiff > zDiff) rx = -ry - rz;
            else if (yDiff > zDiff) ry = -rx - rz;
            else rz = -rx - ry;

            // 4. Find the matching node in our data list
            foreach (var node in nodes)
            {
                if (node.q == rx && node.r == rz) 
                    return node;
            }

            return default;
        }
        public Vector3 GetRandomPositionAtNode(int q, int r)
        {
            // 1. Find the node in our list
            HexNode targetNode = default;
            bool found = false;

            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i].q == q && nodes[i].r == r)
                {
                    targetNode = nodes[i];
                    found = true;
                    break;
                }
            }

            // 2. If node doesn't exist, return zero or a warning
            if (!found)
            {
                Debug.LogWarning($"Attempted to get random position at non-existent hex: {q}, {r}");
                return Vector3.zero;
            }

            // 3. Calculate a random offset within the hex
            // We use the inner radius (distance to edges) to keep the point inside the hex
            // Multiplying by 0.8f ensures we stay away from the very edges/seams
            float range = HexMetrics.innerRadius * 0.8f;
            
            float offsetX = Random.Range(-range, range);
            float offsetZ = Random.Range(-range, range);

            // 4. Return the center world position plus the random jitter
            return new Vector3(
                targetNode.worldPosition.x + offsetX,
                targetNode.worldPosition.y, // Keeps the height of the hex
                targetNode.worldPosition.z + offsetZ
            );            
        }
        public HexNode GetRandomNode()
        {
            if (nodes.Count == 0) return default;
            return nodes[Random.Range(0, nodes.Count)];
        }
        public void SetNodeWalkable(int q, int r, bool walkable)
        {
            int index = nodes.FindIndex(n => n.q == q && n.r == r);
            if (index != -1)
            {
                HexNode node = nodes[index];
                node.isWalkable = walkable;
                nodes[index] = node;
            }
        }

    #endregion
}
