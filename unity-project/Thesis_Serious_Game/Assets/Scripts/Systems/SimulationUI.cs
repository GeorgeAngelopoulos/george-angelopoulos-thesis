using TMPro;
using UnityEngine;

public class SimulationUI : MonoBehaviour
{
    public TextMeshProUGUI timerText;

    void Update()
    {
        float time = SimulationManager.Instance.GetSimulationTime();

        int seconds = Mathf.FloorToInt(time);

        int minutes = seconds / 60;
        int remainingSeconds = seconds % 60;

        timerText.text = $"{minutes:00}:{remainingSeconds:00}";
    }
}
