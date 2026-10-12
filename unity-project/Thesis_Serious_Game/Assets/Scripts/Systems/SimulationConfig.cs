using UnityEngine;

public static class SimulationConfig
{
    // =============================
    // GLOBAL SETTINGS
    // =============================
    #region Global Constants

    public const float BETA = 2.0f;
    public const float LEADER_SHARE = 0.2f;
    public const float STARTING_FOOD = 100.0f;
    public const float POP_UPKEEP_RATE = 1.08f;
    public const float POP_GROWTH_RATE = 0.35f;
    public const int POP_SOFT_CAP = 400;
    public const int PREY_TICKS_PER_ROAM = 6;
    public const int MAX_PREY_PER_EXPLORED_HEX = 2;
    public const float C_MOVE = 5.0f;
    public const float C_HEX_EXPLORE = 200f;

    #endregion

    #region Predator Creation Constants

        public const float MERIT_COST_PER_POINT = 10f;
        public const float DEMAND_COST_PER_POINT = 10f;
        public const float EFFORT_COST_PER_POINT = 10f;

    #endregion

    #region Prey Constants
        
        public const int HARE_WEIGHT = 60;
        public const int STAG_WEIGHT = 30;
        public const int MAMMOTH_WEIGHT = 10;

    #endregion

}
