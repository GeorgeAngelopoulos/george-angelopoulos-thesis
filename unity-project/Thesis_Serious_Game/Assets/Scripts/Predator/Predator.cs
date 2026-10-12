using System.Collections.Generic;
using UnityEngine;

public class Predator : Agent, ISelectable
{
    [Header("Type Variables")]
    public string agentType;
    public float merit;
    public float baseDemand;
    public float huntCost;

    public string tactic;

    [Header("Dynamic Variables")]
    public float food;
    public bool hasActed = false;

    public Decision decision;
    
    // --- Initialization ---
    public void InitializePredator(int id, int q, int r, PredatorBuildData build)
    {
        base.Initialize(id, q, r);

        agentType = build.agentType;
        merit = build.merit;
        baseDemand = build.baseDemand;
        huntCost = build.huntCost;
        food = SimulationConfig.STARTING_FOOD;
        tactic = build.tactic;

    }

    public void OnClick(Vector3 hitPoint)
    {
        GameObject selected = this. gameObject;
        SelectionMenu panel = UIManager.Instance.selectContent.GetComponent<SelectionMenu>();
        panel.selectedObject = selected;

        panel.ShowSelectContent(SelectionMenu.SelectState.Predator);
        UIManager.Instance.onChangeDashboardState(3);
    }
}
