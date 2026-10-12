using UnityEngine;
using System.Collections.Generic;
public class LiveAgentMotor : MonoBehaviour
{
    [Header("Movement Settings")]
    public float rotationSpeed = 10f;
    
    [Header("Surface Alignment")]
    public LayerMask groundLayer;   
    public float raycastDistance = 5f;
    public float hoverOffset = 0.02f; 

    private Vector3 targetWorldPosition;
    private bool hasTarget = false;
    private float calculatedMoveSpeed = 4f; 

    void Start()
    {
        AlignWithSurface();
    }

    void Update()
    {
        // Safety check: Make sure TickManager exists in the scene
        if (TickManager.Instance == null) return;

        // Pull the custom delta time step from your specific TickManager logic
        float simDeltaTime = TickManager.Instance.GetSimDeltaTime();

        // If the game is paused (simDeltaTime == 0), the motor instantly freezes here!
        if (!hasTarget || simDeltaTime <= 0f) return;

        // 1. Horizontal movement calculation
        Vector3 currentPos = transform.position;
        Vector3 horizontalTarget = new Vector3(targetWorldPosition.x, currentPos.y, targetWorldPosition.z);
        
        // Move towards target using the simulation time step
        transform.position = Vector3.MoveTowards(currentPos, horizontalTarget, calculatedMoveSpeed * simDeltaTime);

        // 2. Rotate toward direction using simulation time step
        Vector3 direction = horizontalTarget - currentPos;
        direction.y = 0;
        if (direction.sqrMagnitude > 0.001f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(direction.normalized);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed * simDeltaTime);
        }

        // 3. Keep feet firmly locked onto the hex mesh triangles
        AlignWithSurface();

        // 4. Arrived check
        if (Vector3.Distance(new Vector3(transform.position.x, 0, transform.position.z), 
                             new Vector3(targetWorldPosition.x, 0, targetWorldPosition.z)) < 0.05f)
        {
            hasTarget = false;
        }
    }

    public void SetNewDestination(Vector3 destination)
    {
        if (TickManager.Instance == null) return;

        targetWorldPosition = destination;
        
        // Calculate the straight-line horizontal distance to traverse
        Vector3 currentPosFlat = new Vector3(transform.position.x, 0, transform.position.z);
        Vector3 targetPosFlat = new Vector3(destination.x, 0, destination.z);
        float distance = Vector3.Distance(currentPosFlat, targetPosFlat);

        // Get the tick duration (e.g. 2.0f) from your TickManager
        float totalTimeForTick = TickManager.Instance.TickDuration;

        // Give them a 0.1-second safety window to guarantee they cross the finish line
        // before the next logic tick triggers.
        float buffer = 0.1f;
        float operationalTime = Mathf.Max(0.1f, totalTimeForTick - buffer);
        
        // Speed = Distance / Time
        calculatedMoveSpeed = distance / operationalTime;
        
        hasTarget = true;
    }

    private void AlignWithSurface()
    {
        Vector3 rayOrigin = transform.position + Vector3.up * 2f;
        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, raycastDistance, groundLayer))
        {
            Vector3 adjustedPosition = transform.position;
            adjustedPosition.y = hit.point.y + hoverOffset;
            transform.position = adjustedPosition;
        }
    }
}