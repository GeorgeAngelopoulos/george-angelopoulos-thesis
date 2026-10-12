using UnityEngine;
using System;

public class TickManager : MonoBehaviour
{
    public static TickManager Instance { get; private set; }

    [Header("Tick Settings")]
    [SerializeField] private float tickDuration = 2f;
    [SerializeField] private int maxTicksPerFrame = 3;

    [Header("Controls")]
    [SerializeField] private float timeScale = 1f;
    [SerializeField] private bool isPaused = false;

    private float simulationTime = 0f;
    private int currentTick = 0;

    public int ticksPerDay = 24;

    public int CurrentTick => currentTick;
    public int CurrentDay => currentTick / ticksPerDay + 1;
    public int CurrentHour => currentTick % ticksPerDay;

    public event Action<int> OnTick;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }

    public void Initialize() { Debug.Log("TickManager Initialized"); }

    private void Update()
    {
        if (isPaused) return;

        simulationTime += Time.deltaTime * timeScale;

        int targetTick = Mathf.FloorToInt(simulationTime / tickDuration);

        int processed = 0;

        while (currentTick < targetTick && processed < maxTicksPerFrame)
        {
            currentTick++;
            OnTick?.Invoke(currentTick);
            processed++;
        }
    }

    public void TogglePause(bool value) => isPaused = value;
    public void SetSpeed(float speed) => timeScale = speed;

    public float GetSimulationTime() => simulationTime;

    public float GetSimDeltaTime()
    {
        if (isPaused) return 0f;
        return Time.deltaTime * timeScale;
    }

    // Expose your tickDuration safely via a public property
    public float TickDuration => tickDuration;
}