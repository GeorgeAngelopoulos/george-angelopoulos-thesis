using System.Security.Cryptography.X509Certificates;
using UnityEngine;

public class Prey : Agent, ISelectable
{
    [Header("Type")]
    public PreyType preyType;

    [Header("Type Variables (Auto-loaded from Database)")]
    public float pSolo;
    public float reward;
    public float baseBeta;

    [HideInInspector]
    public float beta;

    [Header("Dynamic Variables")]
    public bool targeted = false;

    // --- Initialization ---
    public void InitializePrey(int id, int q, int r, PreyType type)
    {
        base.Initialize(id, q, r);

        this.preyType = type;

        isAlive = true;
        targeted = false;

        if (PreyDatabase.PREY_PROFILES.TryGetValue(this.preyType, out var profile))
        {
            this.pSolo = profile.pSolo;
            this.reward = profile.reward;
            this.baseBeta = profile.baseBeta;
        }
        else
        {
            Debug.LogError($"PreyType {this.preyType} missing from PreyDatabase profiles!");
        }

        beta = baseBeta * SimulationConfig.BETA;
    }

    public void OnClick(Vector3 hitPoint)
    {
        GameObject selected = this. gameObject;
        SelectionMenu panel = UIManager.Instance.selectContent.GetComponent<SelectionMenu>();
        panel.selectedObject = selected;

        panel.ShowSelectContent(SelectionMenu.SelectState.Prey);
        UIManager.Instance.onChangeDashboardState(3);
    }
}
