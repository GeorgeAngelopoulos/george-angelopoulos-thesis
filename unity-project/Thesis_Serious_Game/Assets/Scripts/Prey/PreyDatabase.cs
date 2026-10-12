using System.Collections.Generic;
using System.Collections.ObjectModel;

public static class PreyDatabase
{
    // Map each PreyType to a named Tuple holding its specific float attributes
    public static readonly ReadOnlyDictionary<PreyType, (float pSolo, float reward, float baseBeta)> PREY_PROFILES;

    static PreyDatabase()
    {
        var profiles = new Dictionary<PreyType, (float pSolo, float reward, float baseBeta)>()
        {
            // PreyType.Type   -> ( pSolo, reward, baseBeta )
            { PreyType.Hare,  (0.7f, 20.0f, 1f)},
            { PreyType.Stag,   (0.4f, 55.0f, 1.5f  )},
            { PreyType.Mammoth, (0.1f, 160.0f, 2.5f)}
        };

        PREY_PROFILES = new ReadOnlyDictionary<PreyType, (float pSolo, float reward, float baseBeta)>(profiles);
    }
}
