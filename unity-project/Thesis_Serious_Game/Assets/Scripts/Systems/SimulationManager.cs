using UnityEngine;
using System;
using UnityEngine.InputSystem;

public class SimulationManager : MonoBehaviour
{
    public static SimulationManager Instance;

    [Header("Simulation Time")]
    public float simulationTime = 0f;

    [Header("Tick Settings")]
    public float tickDuration = 1f; // 1 second per tick

    [Header("Controls")]
    public float timeScale = 1f;
    public bool isPaused = false;

    private int currentTick = 0;

    public event Action<int> OnTick;

    void Awake()
    {
        Instance = this;
    }

    void Update()
    {
        HandleInput();

        if (isPaused) return;

        // Advance time like a stopwatch
        simulationTime += Time.deltaTime * timeScale;

        int newTick = Mathf.FloorToInt(simulationTime / tickDuration);

        // Fire ticks every "second"
        while (currentTick < newTick)
        {
            currentTick++;
            RunTick(currentTick);
        }
    }

    void RunTick(int tick)
    {
        Debug.Log($"TICK {tick} (Time: {simulationTime:F2}s)");
        OnTick?.Invoke(tick);
    }

    // 🔹 Public getters (important!)
    public int GetCurrentTick() => currentTick;
    public float GetSimulationTime() => simulationTime;

    // 🔹 Controls
    public void TogglePause() => isPaused = !isPaused;
    public void SetSpeed(float speed) => timeScale = speed;

    void HandleInput()
    {
        if (Keyboard.current.spaceKey.wasPressedThisFrame)
            TogglePause();

        if (Keyboard.current.digit1Key.wasPressedThisFrame)
            SetSpeed(1f);

        if (Keyboard.current.digit2Key.wasPressedThisFrame)
            SetSpeed(2f);

        if (Keyboard.current.digit3Key.wasPressedThisFrame)
            SetSpeed(5f);
    }

}
