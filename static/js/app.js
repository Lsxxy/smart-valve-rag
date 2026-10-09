/* 问答页启动入口：绑定交互事件，加载智能阀应用配置并建立首个会话。 */
document.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  try {
    await loadApplication();
  } catch {
    renderWelcome('智能阀应用配置加载失败，请检查后端服务。');
    return;
  }
  try {
    await createNewSession();
  } catch (error) {
    els.sessionInfo.textContent = `会话创建失败：${error.message || '请稍后重试'}`;
    renderWelcome('应用已加载，会话暂不可用。');
    updateSideStats();
    return;
  }
  renderWelcome();
});

function bindEvents() {
  els.newSessionBtn.addEventListener('click', createNewSession);
  els.sidebarNewSessionBtn.addEventListener('click', createNewSession);
  els.clearHistoryBtn.addEventListener('click', clearHistory);
  els.sendBtn.addEventListener('click', () => state.inProgress ? cancelStream() : sendMessage());
  [els.sourceFilter, els.roleSelect].forEach(item => {
    item.addEventListener('change', updateScopeDisplay);
    item.addEventListener('input', updateScopeDisplay);
  });
  els.chatInput.addEventListener('input', autoResizeInput);
  els.chatInput.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!state.inProgress) sendMessage();
    }
  });
}
