using UnityEngine;
using UnityEngine.InputSystem;

public class CameraController : MonoBehaviour
{   
    [Header("References")]
    public Camera cam;

    [Header("Movement")]
    public float moveSpeed = 200f;
    public float sprintMultiplier = 2f;
    public float edgeSize = 25f;
    public float smoothTime = 0.005f;

    [Header("Zoom")]
    public float zoomLevel = 20f;
    public float minZoom = 10f;
    public float maxZoom = 40f;
    public float zoomStep = 3f;
    public float zoomSmooth = 10f;

    [Header("Drag Pan")]
    public float dragSpeed = 25f;

    [Header("Bounds")]
    public Vector2 minBounds = new Vector2(-100, -100);
    public Vector2 maxBounds = new Vector2(100, 100);

    private Vector3 targetPosition;
    private Vector3 velocity;

    private Vector2 lastMousePos;
    private bool dragging = false;

    void Start()
    {
        targetPosition = transform.position;

        if (cam == null)
            cam = Camera.main;
    }

    void Update()
    {
        HandleMovement();
        HandleZoom();
        //HandleDrag();
        ClampTarget();

        transform.position = targetPosition;
    }

    void HandleMovement()
    {

        if (Mouse.current == null) return;

            // Ignore when cursor is not inside game window
            Vector2 mousePos = Mouse.current.position.ReadValue();

            if (mousePos.x < 0 || mousePos.y < 0 ||
                mousePos.x > Screen.width || mousePos.y > Screen.height)
            {
                return;
        }

        if (Keyboard.current == null) return;

        Vector3 dir = Vector3.zero;

        // -----------------------
        // KEYBOARD INPUT (WASD)
        // -----------------------
        if (Keyboard.current.wKey.isPressed || Keyboard.current.upArrowKey.isPressed)
            dir += Vector3.forward;

        if (Keyboard.current.sKey.isPressed || Keyboard.current.downArrowKey.isPressed)
            dir += Vector3.back;

        if (Keyboard.current.aKey.isPressed || Keyboard.current.leftArrowKey.isPressed)
            dir += Vector3.left;

        if (Keyboard.current.dKey.isPressed || Keyboard.current.rightArrowKey.isPressed)
            dir += Vector3.right;

        // -----------------------
        // ANALOG EDGE SCROLL
        // -----------------------
        if (Mouse.current != null)
        {
            Vector2 dirInput = Vector2.zero;

            // Horizontal Movement (Left/Right Edges)
            if (mousePos.x < edgeSize)
            {
                // Smoothly ramp from 0 at the edge boundary to 1 at the very pixel 0
                dirInput.x = -Mathf.InverseLerp(edgeSize, 0, mousePos.x);
            }
            else if (mousePos.x > Screen.width - edgeSize)
            {
                dirInput.x = Mathf.InverseLerp(Screen.width - edgeSize, Screen.width, mousePos.x);
            }

            // Vertical Movement (Top/Bottom Edges)
            if (mousePos.y < edgeSize)
            {
                dirInput.y = -Mathf.InverseLerp(edgeSize, 0, mousePos.y);
            }
            else if (mousePos.y > Screen.height - edgeSize)
            {
                dirInput.y = Mathf.InverseLerp(Screen.height - edgeSize, Screen.height, mousePos.y);
            }

            // Normalization prevents "Turbo Diagonal" movement
            if (dirInput.sqrMagnitude > 0.1f)
            {
                // Use normalized if you want constant speed, 
                // or just dirInput if you want pressure-sensitive speed
                Vector3 movement = new Vector3(dirInput.x, 0f, dirInput.y);
                dir += movement; 
            }
        }

        // No input → stop
        if (dir.sqrMagnitude < 0.001f)
            return;

        // -----------------------
        // SPEED
        // -----------------------
        float speed = moveSpeed;

        if (Keyboard.current.leftShiftKey.isPressed)
            speed *= sprintMultiplier;

        // Zoom-based scaling (important for RTS feel)
        float zoomFactor = Mathf.InverseLerp(minZoom, maxZoom, zoomLevel);
        speed *= Mathf.Lerp(1f, 4f, zoomFactor);

        // -----------------------
        // APPLY MOVEMENT
        // -----------------------
        targetPosition += dir * speed * Time.deltaTime;
    }

    void HandleZoom()
    {
        if (Mouse.current == null || cam == null) return;

        float scroll = Mouse.current.scroll.ReadValue().y;

        if (scroll > 0)
            zoomLevel -= zoomStep;

        if (scroll < 0)
            zoomLevel += zoomStep;

        zoomLevel = Mathf.Clamp(zoomLevel, minZoom, maxZoom);

        Vector3 pos = cam.transform.localPosition;

        pos.y = Mathf.Lerp(pos.y, zoomLevel, zoomSmooth * Time.deltaTime);
        pos.z = Mathf.Lerp(pos.z, -zoomLevel, zoomSmooth * Time.deltaTime);

        cam.transform.localPosition = pos;  
    }

    void HandleDrag()
    {
        if (Mouse.current == null) return;

        if (Mouse.current.middleButton.wasPressedThisFrame)
        {
            dragging = true;
            lastMousePos = Mouse.current.position.ReadValue();
        }

        if (Mouse.current.middleButton.wasReleasedThisFrame)
        {
            dragging = false;
        }

        if (!dragging) return;

        Vector2 currentPos = Mouse.current.position.ReadValue();
        Vector2 delta = currentPos - lastMousePos;

        Vector3 move = new Vector3(-delta.x, 0f, -delta.y) * dragSpeed * Time.deltaTime;

        targetPosition += move;

        lastMousePos = currentPos;        
    }

    void ClampTarget()
    {
        targetPosition.x = Mathf.Clamp(targetPosition.x, minBounds.x, maxBounds.x);
        targetPosition.z = Mathf.Clamp(targetPosition.z, minBounds.y, maxBounds.y);        
    }
    
}
