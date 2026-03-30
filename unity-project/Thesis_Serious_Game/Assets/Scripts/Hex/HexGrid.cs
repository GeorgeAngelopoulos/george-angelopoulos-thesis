using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HexGrid : MonoBehaviour
{
    public int width = 5;
	public int height = 5;

	public HexNode nodePrefab;
	public TextMeshProUGUI nodeLabelPrefab;

	Canvas gridCanvas;
	HexMesh hexMesh;


	HexNode[] nodes;

	void Awake () {
		gridCanvas = GetComponentInChildren<Canvas>();
		hexMesh = GetComponentInChildren<HexMesh>();
		nodes = new HexNode[height * width];

		for (int z = 0, i = 0; z < height; z++) {
			for (int x = 0; x < width; x++) {
				CreateNode(x, z, i++);
			}
		}
	}

	void Start()
	{
		hexMesh.Triangulate(nodes);
	}

	void CreateNode(int x, int z, int i)
	{
		Vector3 position;
		position.x = (x + z * 0.5f - z / 2) * (HexMetrics.innerRadius * 2f);
		position.y = 0f;
		position.z = z * (HexMetrics.outerRadius * 1.5f);

		HexNode node = nodes[i] = Instantiate<HexNode> (nodePrefab);
		node.transform.SetParent(transform, false);
		node.transform.localPosition = position;

		TextMeshProUGUI label = Instantiate<TextMeshProUGUI> (nodeLabelPrefab);
		label.rectTransform.SetParent (gridCanvas.transform, false);
		label.rectTransform.anchoredPosition = 
			new Vector2(position.x, position.z);
		label.text = x.ToString() + "\n" + z.ToString();
	}


    
}
