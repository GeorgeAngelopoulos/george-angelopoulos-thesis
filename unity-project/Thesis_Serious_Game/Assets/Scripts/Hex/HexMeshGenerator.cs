using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class HexMeshGenerator : MonoBehaviour, ISelectable
{
    private Mesh hexMesh;
    private List<Vector3> vertices = new List<Vector3>();
    private List<int> triangles = new List<int>();
    private List<Color> colors = new List<Color>();  // Temp for per coloring tile

    private Color walkableGreen = Color.HSVToRGB(0.35f, 0.65f, 0.75f);
    private Color nonWalkableBrown = Color.HSVToRGB(0.08f, 0.55f, 0.60f);


    void Awake()
    {
        GetComponent<MeshFilter>().mesh = hexMesh = new Mesh();
        hexMesh.name = "Hex Mesh";
    }

    public void GenerateMesh(HexNode[] nodes)
    {
        hexMesh.Clear();
        vertices.Clear();
        triangles.Clear();
        colors.Clear();

        for (int i = 0; i < nodes.Length; i++)
        {
            AddHexMesh(nodes[i]);
        }

        hexMesh.vertices = vertices.ToArray();
        hexMesh.triangles = triangles.ToArray();
        hexMesh.colors = colors.ToArray(); // Use this for highlights/terrain types
        hexMesh.RecalculateNormals();
        
        // Update collider so raycasting works on the new shape
        if(TryGetComponent(out MeshCollider collider))
        {
            collider.sharedMesh = hexMesh;
        }
    }

    public void UpdateMeshColors(HexNode[] nodes)
    {
        colors.Clear();

        // Loop through the nodes exactly the same way as generation
        for (int i = 0; i < nodes.Length; i++)
        {
            HexNode node = nodes[i];
                        
            Color tileColor = node.isWalkable ? walkableGreen : nonWalkableBrown;

            // Remember: each node has 6 sides, and each side added 9 vertex colors
            for (int side = 0; side < 6; side++)
            {
                for (int n = 0; n < 9; n++)
                {
                    colors.Add(tileColor);
                }
            }
        }

        // Re-upload the newly updated color list to the GPU
        hexMesh.colors = colors.ToArray();
    }

    void AddHexMesh(HexNode node)
    {
        Vector3 center = node.worldPosition;
        float solidFactor = 0.97f; // 5% gap between tiles
        float thickness = 1.0f; // How "deep" the ground goes

        Color tileColor = node.isWalkable ? walkableGreen : nonWalkableBrown;

        // A hex is made of 6 triangles
        for (int i = 0; i < 6; i++)
        {

            // Top Vertices
            Vector3 v1 = center;
            Vector3 v2 = center + HexMetrics.corners[i] * solidFactor;
            Vector3 v3 = center + HexMetrics.corners[i + 1]* solidFactor;

            // Bottom Vertices
            Vector3 v1Down = v1 - Vector3.up * thickness;
            Vector3 v2Down = v2 - Vector3.up * thickness;
            Vector3 v3Down = v3 - Vector3.up * thickness;

            // 1. Draw Top Face (Walkable surface)
            AddTriangle(v1, v2, v3);

            // 2. Draw Side Walls (Prevents things from seeing "under" the map)
            AddTriangle(v2, v2Down, v3);
            AddTriangle(v3, v2Down, v3Down);
            
            // 3. Apply color to all 9 vertices generated in this step
            for(int n = 0; n < 9; n++) 
            {
                colors.Add(tileColor);
            }
        }
    }
    void AddTriangle(Vector3 v1, Vector3 v2, Vector3 v3)
    {
        int vertexIndex = vertices.Count;
        vertices.Add(v1);
        vertices.Add(v2);
        vertices.Add(v3);

        triangles.Add(vertexIndex);
        triangles.Add(vertexIndex + 1);
        triangles.Add(vertexIndex + 2);
    }
    
    public void OnClick(Vector3 hitPoint)
    {
        HexNode selected = HexGridManager.Instance.GetNodeAtPoint(hitPoint);
        SelectionMenu panel = UIManager.Instance.selectContent.GetComponent<SelectionMenu>();
        panel.selectedNode = selected;
        panel.selectedObject = null;

        panel.ShowSelectContent(SelectionMenu.SelectState.Hex);
        UIManager.Instance.onChangeDashboardState(3);
    }
}
