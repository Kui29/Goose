//! What a provider call is for, carried from the caller to the provider on the same task, as
//! goose's `session_context` carries the session id.

tokio::task_local! {
    static REPLACES_HISTORY: bool;
    static SESSION: Option<String>;
    static TOOL_CALLS_ONLY: bool;
}

/// Runs `f` as a call whose answer replaces the history of the conversation it is about, as a
/// compaction's summary does: what that conversation held is not asked about again as it stands,
/// so a provider keeping it for a later turn can let it go.
pub async fn replacing_history<F: std::future::Future>(f: F) -> F::Output {
    REPLACES_HISTORY.scope(true, f).await
}

/// Whether the current call replaces its conversation's history; `false` outside
/// [`replacing_history`]. Read it before handing work to another task: a task-local does not
/// follow `tokio::spawn`.
pub fn replaces_history() -> bool {
    REPLACES_HISTORY
        .try_with(|replaces| *replaces)
        .unwrap_or(false)
}

/// Runs `f` as a call made for the conversation `session_id`. goose's `with_session_id` scopes
/// every provider call this way, so a provider keeping one conversation between calls can tell
/// whose it is: a compaction's summary replaces the history of the conversation it summarises,
/// not of whichever conversation the provider happens to hold.
pub async fn for_session<F: std::future::Future>(session_id: Option<String>, f: F) -> F::Output {
    SESSION.scope(session_id, f).await
}

/// The conversation the current call is made for; `None` outside [`for_session`]. Like
/// [`replaces_history`], read it before handing work to another task.
pub fn session() -> Option<String> {
    SESSION.try_with(|session| session.clone()).ok().flatten()
}

/// Runs `f` as a call whose answer is used only if it calls a tool, as a completeness check's
/// round is: the agent hides and drops its prose. A provider that can end a generation without
/// losing what it keeps may stop at the first text that comes before any tool call.
pub async fn tool_calls_only<F: std::future::Future>(f: F) -> F::Output {
    TOOL_CALLS_ONLY.scope(true, f).await
}

/// Whether the current call is [`tool_calls_only`]; `false` outside it. Read it before handing
/// work to another task.
pub fn wants_tool_calls_only() -> bool {
    TOOL_CALLS_ONLY.try_with(|only| *only).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn only_a_call_made_inside_the_scope_replaces_history() {
        assert!(!replaces_history());
        assert!(replacing_history(async { replaces_history() }).await);
        assert!(!replaces_history(), "the scope ends with its future");
    }

    #[tokio::test]
    async fn a_spawned_task_does_not_inherit_it() {
        let inherited =
            replacing_history(async { tokio::spawn(async { replaces_history() }).await })
                .await
                .unwrap();
        assert!(!inherited, "read it before spawning");
    }

    #[tokio::test]
    async fn only_a_call_made_inside_the_scope_wants_tool_calls_only() {
        assert!(!wants_tool_calls_only());
        assert!(tool_calls_only(async { wants_tool_calls_only() }).await);
        assert!(!wants_tool_calls_only(), "the scope ends with its future");
    }

    #[tokio::test]
    async fn a_call_knows_the_conversation_it_is_made_for() {
        assert_eq!(session(), None);
        let inner = for_session(Some("chat".to_string()), async {
            let summary = for_session(Some("older".to_string()), async { session() }).await;
            (summary, session())
        })
        .await;
        assert_eq!(
            inner,
            (Some("older".to_string()), Some("chat".to_string())),
            "the innermost scope wins, and ends with its future"
        );
        assert_eq!(session(), None);
    }
}
