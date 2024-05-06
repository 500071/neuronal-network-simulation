# SIMULACE NEURONÁLNÍ SÍTĚ
***

## Popis funkcí jednotlivých souborů

**Pomocné kódy**

`grafy_site` obsahuje kódy ke generování matic souslednosti pro všechny typy sítí a jejich vizualizaci (obrázky v kapitole 2)


**Modely neuronů**

Všechny mají jako výstup graf napětí na čase (spřažené v součtu i jednotlivé) a periodogram
Všechny používájí Runge Kutta 45, Euler-Maruyama bude další krok.

`interneuron_1` simulace jednoho interneuronu

`interneuron_coupled` simulace n spřažených interneuronů

`interneuron_coupled_clustered` simulace n spřažených interneuronů ve shlucích **TODO**

`ML_coupled` simulace n spřažených Morris-Lecar modelů

`DP_1` simulace jednoho DP neuronu

`DP_coupled` simulace n spřažených DP neuronů

**Webová aplikace**

`app` kostra pro webové rozhraní

`home` stránka s úvodními informacemi o aplikace

`info` informace o modelech

`ML` simulace Morris-Lecar modelu

`IN` simulace modelu interneuronu (HOTOVO - chybí periodogram)

`DP` simulace Destexhe-Paré modelu

`ML_coupled` simulace více shluků Morris-Lecar modelu

`IN_coupled` simulace více shluků modelu interneuronu

`DP_coupled` simulace více shluků Destexhe-Paré modelu
