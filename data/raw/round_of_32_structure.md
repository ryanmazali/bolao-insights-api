# Estrutura oficial do Round of 32 — Copa do Mundo 2026
Fonte: FIFA Tournament Regulations, Annex C (via Wikipedia, conferido em 17/06/2026)
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage

## Os 16 jogos do Round of 32 (posições fixas — 8 são diretas, 8 dependem dos 3ºs)

| Jogo | Time A | Time B |
|---|---|---|
| 73 | Runner-up Group A | Runner-up Group B |
| 74 | Winner Group E | Best 3rd place Group A/B/C/D/F |
| 75 | Winner Group F | Runner-up Group C |
| 76 | Winner Group C | Runner-up Group F |
| 77 | Winner Group I | Best 3rd place Group C/D/F/G/H |
| 78 | Runner-up Group E | Runner-up Group I |
| 79 | Winner Group A | Best 3rd place Group C/E/F/H/I |
| 80 | Winner Group L | Best 3rd place Group E/H/I/J/K |
| 81 | Winner Group D | Best 3rd place Group B/E/F/I/J |
| 82 | Winner Group G | Best 3rd place Group A/E/H/I/J |
| 83 | Runner-up Group K | Runner-up Group L |
| 84 | Winner Group H | Runner-up Group J |
| 85 | Winner Group B | Best 3rd place Group E/F/G/I/J |
| 86 | Winner Group J | Runner-up Group H |
| 87 | Winner Group K | Best 3rd place Group D/E/I/J/L |
| 88 | Runner-up Group D | Runner-up Group G |

Observação: 8 jogos (73, 75, 76, 78, 83, 84, 86, 88) são 100% determinados
pelos resultados da fase de grupos (1ºs e 2ºs colocados), independente de
quais terceiros se classificam.

Os outros 8 jogos (74, 77, 79, 80, 81, 82, 85, 87) têm o lado direito
("Best 3rd place Group X/Y/Z/W/V") que só é resolvido DEPOIS de sabermos
quais 8 grupos tiveram seu 3º colocado classificado entre os 8 melhores.

## Como resolver os 8 jogos de terceiros (lógica do Anexo C)

A FIFA pré-definiu, para cada uma das 495 combinações possíveis de "quais
8 grupos (de 12) tiveram seu 3º colocado classificado", qual 3º colocado
específico cai em qual dos 8 jogos. Isso evita que, por exemplo, dois
terceiros do mesmo grupo de origem caiam no mesmo bracket-side de forma
desequilibrada, e tenta evitar repetir confrontos entre seleções do mesmo
grupo já visto na fase de grupos.

Arquivo separado `third_place_combinations.csv` contém as 495 linhas
(uma por combinação), com colunas:
combo_id, group_E_slot(match74), group_C_slot(match77), group_I_slot(match79),
group_L_slot(match80), group_D_slot(match81), group_G_slot(match82),
group_B_slot(match85), group_K_slot(match87)

A lógica de implementação:
1. Depois da fase de grupos, identifique quais 8 grupos (das 12) tiveram
   o 3º colocado entre os 8 melhores (por pontos, saldo, gols pró, etc.)
2. Ordene esses 8 grupos alfabeticamente (ex: "C;D;E;G;H;I;K;L")
3. Procure essa combinação exata na tabela de 495 (campo "groups_combo")
4. A linha encontrada diz exatamente qual 3º colocado vai para qual dos
   8 jogos (74, 77, 79, 80, 81, 82, 85, 87)

## Critérios de desempate para "8 melhores terceiros colocados"
(em ordem de prioridade, conforme regulamento FIFA)
1. Pontos
2. Saldo de gols
3. Gols marcados
4. Maior "team conduct score" (cartões amarelos/vermelhos — menos é melhor)
5. Posição no ranking FIFA

## Critérios de desempate dentro de um grupo (1º/2º/3º/4º)
1. Pontos
2. Confronto direto entre os times empatados (se aplicável)
3. Saldo de gols
4. Gols marcados
5. Team conduct score
6. Ranking FIFA

## Estrutura completa do mata-mata (Round of 32 até a Final)

R32 (16 jogos, 73-88) → R16 (8 jogos, 89-96) → QF (4 jogos, 97-100)
→ SF (2 jogos, 101-102) → 3rd place playoff (103) + Final (104)

Mapeamento R32 → R16:
- Match 89: Winner 74 vs Winner 77
- Match 90: Winner 73 vs Winner 75
- Match 91: Winner 76 vs Winner 78
- Match 92: Winner 79 vs Winner 80
- Match 93: Winner 83 vs Winner 84
- Match 94: Winner 81 vs Winner 82
- Match 95: Winner 86 vs Winner 88
- Match 96: (não capturado explicitamente na fonte — verificar, provável
  Winner 85 vs Winner 87, já que são os 2 jogos de R32 restantes não
  usados em nenhum outro R16 mapeado)

R16 → QF:
- Match 97: Winner 89 vs Winner 90
- Match 98: Winner 93 vs Winner 94
- Match 99: Winner 91 vs Winner 92
- Match 100: Winner 95 vs Winner 96

QF → SF:
- Match 101: Winner 97 vs Winner 98
- Match 102: Winner 99 vs Winner 100

SF → Final/3rd place:
- Match 103 (3rd place): Loser 101 vs Loser 102
- Match 104 (Final): Winner 101 vs Winner 102

## Confirmação do Match 96 (era a única lacuna, agora resolvida)
CONFIRMADO via FIFA.com, NBC Sports, MLS Soccer e Bleacher Report
(4 fontes independentes, 17/06/2026):
  Match 96: Winner Match 85 vs Winner Match 87 — BC Place, Vancouver
  Data: terça-feira, 7 de julho de 2026

Estrutura R32→R16 completa e confirmada:
- Match 89: Winner 74 vs Winner 77 — Philadelphia — sáb 4 jul
- Match 90: Winner 73 vs Winner 75 — Houston — sáb 4 jul
- Match 91: Winner 76 vs Winner 78 — New York/New Jersey — dom 5 jul
- Match 92: Winner 79 vs Winner 80 — Mexico City — dom 5 jul
- Match 93: Winner 83 vs Winner 84 — Dallas — seg 6 jul
- Match 94: Winner 81 vs Winner 82 — Seattle — seg 6 jul
- Match 95: Winner 86 vs Winner 88 — Atlanta — ter 7 jul
- Match 96: Winner 85 vs Winner 87 — Vancouver — ter 7 jul

R16→QF (confirmado):
- Match 97: Winner 89 vs Winner 90 — Boston — qui 9 jul
- Match 98: Winner 93 vs Winner 94 — Los Angeles — sex 10 jul
- Match 99: Winner 91 vs Winner 92 — Miami — qui 9 jul
- Match 100: Winner 95 vs Winner 96 — Kansas City — sáb 11 jul

QF→SF (confirmado):
- Match 101: Winner 97 vs Winner 98 — Dallas — ter 14 jul
- Match 102: Winner 99 vs Winner 100 — Atlanta — qua 15 jul

SF→Final/3rd place (confirmado):
- Match 103 (3º lugar): Loser 101 vs Loser 102 — Miami — sáb 18 jul
- Match 104 (Final): Winner 101 vs Winner 102 — New York/New Jersey — dom 19 jul

ESTRUTURA 100% VERIFICADA — nenhuma lacuna restante.
