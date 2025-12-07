WITH AfiliadosAgrupados AS (
    SELECT
        DocEmp,
        TipoDocEmp,
        NomEmp,
        CodclaseEconomicaEmp,
        CodSeccionEconomicaEmp,
        SeccionEconomicaEmp,
        CASE
            WHEN TipoDocEmp = 'NIT' THEN SUBSTRING(DocEmp, 1, 9)
            ELSE DocEmp
        END AS NitSinDV,
        Periodo,
        SUM(CASE WHEN GeneroAfil = 'Femenino' THEN 1 ELSE 0 END) AS Femenino,
        SUM(CASE WHEN GeneroAfil = 'Masculino' THEN 1 ELSE 0 END) AS Masculino,
        SUM(CASE WHEN GeneroAfil = 'Desconocido' THEN 1 ELSE 0 END) AS Desconocido,
        SUM(CASE WHEN CategoriaAfil = 'A' THEN 1 ELSE 0 END) AS TarifaA,
        SUM(CASE WHEN CategoriaAfil = 'B' THEN 1 ELSE 0 END) AS TarifaB,
        SUM(CASE WHEN CategoriaAfil = 'C' THEN 1 ELSE 0 END) AS TarifaC,
        SUM(CASE WHEN EdadAfil <= 25 THEN 1 ELSE 0 END) AS Jovenes,
        SUM(CASE WHEN EdadAfil > 25 AND EdadAfil <= 40 THEN 1 ELSE 0 END) AS AdultoJoven,
        SUM(CASE WHEN EdadAfil > 40 AND EdadAfil <= 59 THEN 1 ELSE 0 END) AS Adulto,
        SUM(CASE WHEN EdadAfil > 59 THEN 1 ELSE 0 END) AS AdultoMayor,
        COUNT(*) AS TotalAfiliados
    FROM
        [BI_corporativo].[AfiliadosComfama]
    WHERE
        (Periodo BETWEEN '202411' AND '202510')
        AND (TipoEmpresa LIKE 'Empresa 4%' OR TipoEmpresa = 'Empresa')
        AND TipoAfiliado = 'Titular'
        AND NegocioAfil = 'Empresas'
        AND SeccionEconomicaEmp NOT LIKE 'T_Actv%'
        -- Validaciones
        --AND LTRIM(RTRIM(DocEmp)) <> ''
        AND DocEmp IS NOT NULL
        AND DocEmp <> '0'
        AND TipoDocEmp IS NOT NULL
        AND NomEmp IS NOT NULL
        AND SeccionEconomicaEmp IS NOT NULL
    GROUP BY
        DocEmp, TipoDocEmp, NomEmp, CodclaseEconomicaEmp, CodSeccionEconomicaEmp, SeccionEconomicaEmp, Periodo
),
PromediosAfiliados AS (
    SELECT
        DocEmp,
        MAX(TipoDocEmp)              AS TipoDocEmp,
        MAX(NomEmp)                  AS NomEmp,
        MAX(CodclaseEconomicaEmp)    AS CodclaseEconomicaEmp,
        MAX(CodSeccionEconomicaEmp)  AS CodSeccionEconomicaEmp,
        MAX(SeccionEconomicaEmp)     AS SeccionEconomicaEmp,
        MAX(NitSinDV)                AS NitSinDV,

        AVG(CONVERT(decimal(18,4), Femenino))       AS Femenino,
        AVG(CONVERT(decimal(18,4), Masculino))      AS Masculino,
        AVG(CONVERT(decimal(18,4), Desconocido))    AS Desconocido,
        AVG(CONVERT(decimal(18,4), TarifaA))        AS TarifaA,
        AVG(CONVERT(decimal(18,4), TarifaB))        AS TarifaB,
        AVG(CONVERT(decimal(18,4), TarifaC))        AS TarifaC,
        AVG(CONVERT(decimal(18,4), Jovenes))        AS Jovenes,
        AVG(CONVERT(decimal(18,4), AdultoJoven))    AS AdultoJoven,
        AVG(CONVERT(decimal(18,4), Adulto))         AS Adulto,
        AVG(CONVERT(decimal(18,4), AdultoMayor))    AS AdultoMayor,
        AVG(CONVERT(decimal(18,4), TotalAfiliados)) AS TotalAfiliados
    FROM AfiliadosAgrupados
    GROUP BY DocEmp--,TipoDocEmp,NomEmp,CodclaseEconomicaEmp,CodSeccionEconomicaEmp,SeccionEconomicaEmp, NitSinDV
),
SeccionUnica AS (
    SELECT
        DocEmp,
        TipoDocEmp,
        SeccionEconomicaEmp
    FROM [BI_Corporativo].[AportantesComfama]
    WHERE
        Periodo = '202510'
        AND (TipoAportante LIKE 'Empresa 4%' OR TipoAportante = 'Empresa')
        AND NegocioEmp = 'Empresas'
        AND SeccionEconomicaEmp NOT LIKE 'T_Actv%'
        AND EstadoAportante = 'Vigente'
        -- Validaciones
        --AND LTRIM(RTRIM(DocEmp)) <> ''
        AND DocEmp IS NOT NULL
        AND DocEmp <> '0'
        AND TipoDocEmp IS NOT NULL
        AND SeccionEconomicaEmp IS NOT NULL
),
Aportes AS (
    SELECT
        DocEmp,
        ISNULL(AVG(AportePagado), 0) AS Aporte
    FROM [BI_Corporativo].[AportantesComfama]
    WHERE EstadoAportante = 'Vigente'
        AND (TipoAportante LIKE 'Empresa 4%' OR TipoAportante LIKE 'Empresa%')
        AND NegocioEmp = 'Empresas'
        AND (Periodo BETWEEN '202411' AND '202510')
        AND AportePagado > 0
        -- Validaciones
        --AND LTRIM(RTRIM(DocEmp)) <> ''
        AND DocEmp IS NOT NULL
        AND DocEmp <> '0'
    GROUP BY DocEmp
),
UsoServiciosAgrupados AS (
    SELECT
        DocEmp,
        PeriodoUso,
        ISNULL(SUM(Cantidad), 0) AS TotalUsos,
        ISNULL(SUM(CASE WHEN GrupoObjetivo = 'CULTURA'   THEN Cantidad ELSE 0 END), 0) AS CULTURA,
        ISNULL(SUM(CASE WHEN GrupoObjetivo = 'EDUCACION' THEN Cantidad ELSE 0 END), 0) AS EDUCACION,
        ISNULL(SUM(CASE WHEN GrupoObjetivo = 'SALUD'     THEN Cantidad ELSE 0 END), 0) AS SALUD,
        ISNULL(SUM(CASE WHEN GrupoObjetivo = 'EMPLEO'    THEN Cantidad ELSE 0 END), 0) AS EMPLEO
    FROM 
        [Prisma].[AdherenciaUsoServicios]
    WHERE 
        Servicio NOT LIKE '%Cuota monetaria%'
        AND PeriodoUso BETWEEN '202411' AND '202510'
        AND CategoriaAfiliacion IN ('A','B','C')
        AND TipoAfiliado = 'Titular'
        AND DocEmp IS NOT NULL
        AND DocEmp <> '0'
        --AND LTRIM(RTRIM(DocEmp)) <> ''
    GROUP BY 
        DocEmp, PeriodoUso
),
PromediosUsos AS (
    SELECT
        DocEmp,
        AVG(CONVERT(decimal(18,4), TotalUsos))  AS TotalUsos,
        AVG(CONVERT(decimal(18,4), CULTURA))    AS Cultura,
        AVG(CONVERT(decimal(18,4), EDUCACION))  AS Educacion,
        AVG(CONVERT(decimal(18,4), SALUD))      AS Salud,
        AVG(CONVERT(decimal(18,4), EMPLEO))     AS Empleo
    FROM UsoServiciosAgrupados
    GROUP BY DocEmp
),
AfiliadosConSeccionVigente AS (
    SELECT
        p.DocEmp, 
        p.TipoDocEmp,
        p.NomEmp,
        p.CodclaseEconomicaEmp,
        p.CodSeccionEconomicaEmp,
        s.SeccionEconomicaEmp,     -- sección económica vigente (último período)
        p.NitSinDV,
        p.Femenino,
        p.Masculino,
        p.Desconocido,
        p.TarifaA,
        p.TarifaB,
        p.TarifaC,
        p.Jovenes,
        p.AdultoJoven,
        p.Adulto,
        p.AdultoMayor,
        p.TotalAfiliados
    FROM PromediosAfiliados p
    INNER JOIN SeccionUnica s
        ON p.DocEmp = s.DocEmp
),
Maestro AS(
    SELECT
        acs.DocEmp,
        acs.TipoDocEmp,
        acs.NomEmp,
        acs.CodclaseEconomicaEmp,
        acs.CodSeccionEconomicaEmp,
        acs.SeccionEconomicaEmp,
        acs.NitSinDV,

        acs.Femenino,
        acs.Masculino,
        acs.Desconocido,
        acs.TarifaA,
        acs.TarifaB,
        acs.TarifaC,
        acs.Jovenes,
        acs.AdultoJoven,
        acs.Adulto,
        acs.AdultoMayor,
        acs.TotalAfiliados,

        COALESCE(ap.Aporte, 0.0)      AS Aporte,
        COALESCE(pu.TotalUsos, 0)     AS TotalUsos,
        COALESCE(pu.Cultura, 0)       AS UsosCultura,
        COALESCE(pu.Educacion, 0)     AS UsosEducacion,
        COALESCE(pu.Salud, 0)         AS UsosSalud,
        COALESCE(pu.Empleo, 0)        AS UsosEmpleo,

        -- Porcentajes por género
        CAST(acs.Femenino / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeFemenino,
        CAST(acs.Masculino / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeMasculino,
        CAST(acs.Desconocido / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeDesconocido,

        -- Porcentajes por tarifa
        CAST(acs.TarifaA / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeTarifaA,
        CAST(acs.TarifaB / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeTarifaB,
        CAST(acs.TarifaC / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeTarifaC,

        -- Porcentajes por grupo etario
        CAST(acs.Jovenes       / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeJovenes,
        CAST(acs.AdultoJoven   / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeAdultoJoven,
        CAST(acs.Adulto        / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeAdulto,
        CAST(acs.AdultoMayor   / NULLIF(acs.TotalAfiliados, 0) AS DECIMAL(10,3)) AS PorcentajeAdultoMayor

    FROM AfiliadosConSeccionVigente acs
    LEFT JOIN Aportes ap
        ON acs.DocEmp = ap.DocEmp
    LEFT JOIN PromediosUsos pu
        ON acs.DocEmp = pu.DocEmp
    WHERE acs.TotalAfiliados >=2
)

SELECT * 
FROM Maestro;


--La consulta a continuación nos permite validar que no haya empresas repetidas, o bien, 
--filas iguales salvo una minimidad

--SELECT count(*)
--from Maestro

--DocEmpDuplicados AS (
--    SELECT
--        DocEmp
--    FROM Maestro
--    GROUP BY DocEmp
--    HAVING COUNT(*) > 1
--)
--SELECT m.*
--FROM Maestro m
--INNER JOIN DocEmpDuplicados d
--    ON m.DocEmp = d.DocEmp
--ORDER BY m.DocEmp;