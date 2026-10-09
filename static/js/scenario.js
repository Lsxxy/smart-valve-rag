/* 智能阀应用配置：加载唯一应用及其产品资料分类和知识库版本。 */
async function loadApplication() {
  let application;
  try {
    application = await fetchJson('/api/application');
  } catch (error) {
    state.application = null;
    els.scenarioDescription.textContent = `应用配置加载失败：${error.message || '请检查后端服务'}`;
    els.contextTitle.textContent = '智能阀知识助手';
    els.contextSubtitle.textContent = '无法读取应用配置';
    throw error;
  }

  state.application = application;
  state.lastDiagnostics = null;
  state.lastStreamStatus = '等待提问';
  state.lastHitType = '-';
  state.lastSourceCount = 0;
  state.lastTraceId = null;
  els.scenarioDescription.textContent = `${application.industry}｜${application.description}`;
  els.contextTitle.textContent = application.display_name;
  els.contextSubtitle.textContent = application.business_domain;
  await Promise.allSettled([loadSources(), loadKbVersion()]);
  renderSamples();
  updateScopeDisplay();
  updateSideStats();
}

async function loadSources() {
  try {
    const payload = await fetchJson('/api/sources');
    const options = payload.source_options || (payload.sources || []).map(item => ({ value: item, label: item }));
    els.sourceFilter.innerHTML = '<option value="">全部</option>';
    for (const source of options) {
      const option = document.createElement('option');
      option.value = source.value;
      option.textContent = source.label;
      els.sourceFilter.appendChild(option);
    }
  } catch {
    els.sourceFilter.innerHTML = '<option value="">全部</option>';
  }
}

async function loadKbVersion() {
  try {
    const payload = await fetchJson('/api/kb_versions');
    state.kbVersion = payload.effective_active_version || payload.active_version || null;
  } catch {
    state.kbVersion = null;
  }
  els.kbPill.innerHTML = `<i class="fas fa-code-branch"></i><span>${escapeHtml(shortText(state.kbVersion || '未激活', 24))}</span>`;
}

function currentApplication() {
  return state.application;
}
