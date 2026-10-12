using UnityEngine;

public static class HexMetrics
{
    public const float outerRadius = 15f;
    public const float innerRadius = outerRadius * 0.866025404f; // sqrt(3)/2

    // Pointy-topped corners
    public static Vector3[] corners = {
        new Vector3(0f, 0f, outerRadius),
        new Vector3(innerRadius, 0f, 0.5f * outerRadius),
        new Vector3(innerRadius, 0f, -0.5f * outerRadius),
        new Vector3(0f, 0f, -outerRadius),
        new Vector3(-innerRadius, 0f, -0.5f * outerRadius),
        new Vector3(-innerRadius, 0f, 0.5f * outerRadius),
        new Vector3(0f, 0f, outerRadius) // Closing the loop
    };
}
