using UnityEngine;

public class PredatorAppearanceCustomizer : MonoBehaviour
{
    [Header("Anchor Points")]
    [SerializeField] private Transform bodyAnchor;
    [SerializeField] private Transform clothingAnchor;
    [SerializeField] private Transform weaponAnchor;

    [Header("Asset Pools (Element 0, 1, 2)")]
    [SerializeField] private GameObject[] bodyPrefabs;
    [SerializeField] private GameObject[] clothingPrefabs;
    [SerializeField] private GameObject[] weaponPrefabs;

    public void AssembleAppearance(PredatorBuildData data)
    {
        // 1. Determine Asset Array Indices based on your float values
        int weaponIndex = GetTierIndex(data.merit, 1.0f, 2.0f);
        int bodyIndex = GetTierIndex(data.huntCost, 5.0f, 10.0f);
        int clothingIndex = GetTierIndex(data.baseDemand, 0.15f, 0.3f);

        // 2. Spawn the physical parts into their respective anchors
        SpawnPart(bodyPrefabs[bodyIndex], bodyAnchor);
        GameObject spawnedClothes = SpawnPart(clothingPrefabs[clothingIndex], clothingAnchor);
        SpawnPart(weaponPrefabs[weaponIndex], weaponAnchor);

        // 3. Apply the Tactic Hue Tint to the spawned clothes
        if (spawnedClothes != null) { ApplyTacticColor(spawnedClothes, data.tactic); }
    }

    private int GetTierIndex(float value, float lowThreshold, float highThreshold)
    {
        if (value <= lowThreshold) return 0;  // Tier 1 (Weak/Low)
        if (value <= highThreshold) return 1; // Tier 2 (Medium)
        return 2;                             // Tier 3 (High)
    }

    private GameObject SpawnPart(GameObject prefab, Transform anchor)
    {
        if (prefab == null || anchor == null) return null;
        
        // Instantiate the piece directly as a child of the anchor point
        GameObject piece = Instantiate(prefab, anchor.position, anchor.rotation, anchor);
        piece.transform.localPosition = Vector3.zero;
        piece.transform.localRotation = Quaternion.identity;
        return piece;
    }

    private void ApplyTacticColor(GameObject clothingObject, string tactic)
    {
        // Determine color based on your 4 tactics
        Color tacticColor = Color.white;
        switch (tactic)
        {
            case "Egalitarian":  tacticColor = Color.green; break;
            case "Meritocratic": tacticColor = Color.blue;  break;
            case "Altruistic":   tacticColor = Color.red;  break;
            case "Asymmetric":   tacticColor = Color.gold;   break;
        }

        // Search the spawned clothing item for its renderer to swap the material color tint
        Renderer renderer = clothingObject.GetComponentInChildren<Renderer>();
        if (renderer != null)
        {
            // Using .material automatically creates a local instance copy so it doesn't change your project asset files
            renderer.material.color = tacticColor; 
        }
    }
}
