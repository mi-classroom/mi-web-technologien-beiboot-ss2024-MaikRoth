# Review-Prozess

## 1. Branching-Strategie
- Verwendung einer Branching-Strategie. Der `main` oder `master` Branch sollte als stabile Produktionsversion betrachtet werden.
- Für neue Features oder Bugfixes wird ein neuer Branch vom `main` Branch aus erstellt. Der Branch wird nach dem Feature oder Issue, z.B. `feature/login-page` oder `bugfix/fix-crash` bennant.

## 2. Code schreiben
- Das Feature oder der Bug wird im neu erstellten Branch erstellt oder behoben.

## 4. Pull Request (PR) erstellen
- Nachdem die Arbeit an einem Feature oder Bugfix abgeschlossen ist, wird ein Pull Request (PR) vom Feature-Branch zum `main` Branch erstellt.
- Im PR wird eine klare Beschreibung angegeben, was geändert wurde und warum.

## 5. Review durch Buddy
- Der Review Buddy wird als Reviewer zugewiesen.
- Der Buddy überprüft den Code, stellt sicher, dass er den Anforderungen entspricht, keine offensichtlichen Bugs oder Code-Smells enthält und die eventuellen Tests bestehen.
- Der Buddy kann Kommentare hinterlassen und Änderungen vorschlagen.

## 6. Feedback einarbeiten
- Der Autor des PRs arbeitet das Feedback ein und nimmt notwendige Änderungen vor.
- Falls größere Änderungen vorgenommen werden, kann der Review Buddy den PR erneut überprüfen.

## 7. Freigabe und Merge
- Sobald der Review Buddy zufrieden ist und der Code alle Anforderungen erfüllt, gibt er den PR frei.
- Der PR kann dann in den `main` Branch gemerged werden.

## 8. Dokumentation
- Der Review-Prozess wird Dokumentiert, um sicher zu stellen, dass alle Beteiligten ihn kennen und verstehen.

---

## Beispiel-Konfiguration für ein GitHub-Repository

### 1. Branch Protection Rules
- Gehen Sie zu den Einstellungen des Repositories.
- Unter "Branches" -> "Branch protection rules" eine neue Regel für den `main` Branch erstellen.
- Aktivieren Sie "Require pull request reviews before merging" und setzen Sie die Anzahl der erforderlichen Reviews auf 1.
- Optional: Aktivieren Sie "Require status checks to pass before merging" und wählen Sie die relevanten Tests aus.

### 2. Pull Request Template
- Erstellen Sie eine Datei `.github/PULL_REQUEST_TEMPLATE.md` im Repository mit einem Standard-Template für PR-Beschreibungen.

---

## Beispiel für eine PR-Template

```markdown
## Beschreibung
<!-- Bitte beschreiben Sie, was geändert wurde und warum. -->

## Änderungen
<!-- Listen Sie die Änderungen auf, die vorgenommen wurden. -->

- [ ] Änderung 1
- [ ] Änderung 2

## Tests
<!-- Beschreiben Sie, wie die Änderungen getestet wurden. -->

## Relevante Issues
<!-- Verlinken Sie relevante Issues. -->

## Checkliste
- [ ] Code entspricht den Anforderungen
- [ ] Keine offensichtlichen Bugs oder Code-Smells
- [ ] Dokumentation aktualisiert (falls erforderlich)
```